#include "controllers/hybrid_force_motion.h"

#include <iostream>

#include "panda.h"

using namespace controllers;

// clang-format off
double impedance_data[36] = {600, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 30};
// clang-format on
const Eigen::Matrix<double, 6, 6> HybridForceMotion::kDefaultImpedance = Eigen::Matrix<double, 6, 6>(impedance_data);
const double HybridForceMotion::kDefaultDampingRatio = 1.0;
const double HybridForceMotion::kDefaultNullspaceStiffness = 0.5;
const double HybridForceMotion::kDefaultForceProportionalGain = 1.0;
const double HybridForceMotion::kDefaultForceIntegralGain = 2.0;
const Eigen::Matrix<bool, 6, 1> HybridForceMotion::kDefaultSelection = (Eigen::Matrix<bool, 6, 1>() << false, false, false, false, false, false).finished();
const double HybridForceMotion::kDefaultFilterCoeff = 1.0;
const double HybridForceMotion::kDefaultDqThreshold = 1e-3;

HybridForceMotion::HybridForceMotion(
    std::shared_ptr<motion::CartesianTrajectory> trajectory,
    const Vector7d& q_init,
    const Eigen::Matrix<double, 6, 6>& impedance,
    const double& damping_ratio,
    const double& nullspace_stiffness,
    const double& force_k_p,
    const double& force_k_i,
    const Eigen::Matrix<bool, 6, 1>& selection,
    const double& filter_coeff)
    : traj_(trajectory),
      q_init_(q_init),
      dq_threshold_(kDefaultDqThreshold) {
  K_p_ = impedance;
  K_p_target_ = impedance;
  damping_ratio_ = damping_ratio;
  _computeDamping();
  K_d_ = K_d_target_;
  nullspace_stiffness_ = nullspace_stiffness;
  nullspace_stiffnes_target_ = nullspace_stiffness;
  k_p_ = force_k_p;
  k_p_target_ = force_k_p;
  k_i_ = force_k_i;
  k_i_target_ = force_k_i;
  selection_ = selection;
  selection_target_ = selection;
  filter_coeff_ = filter_coeff;
}

void HybridForceMotion::_computeDamping() {
  K_d_target_ = damping_ratio_ * 2 * K_p_.cwiseSqrt();
}

franka::Torques HybridForceMotion::step(const franka::RobotState& robot_state, franka::Duration& duration) {
  auto position = traj_->getPosition(getTime());
  auto orientation = traj_->getOrientation(getTime());
  // In a real scenario, the target force would likely come from a trajectory or user input
  Eigen::Matrix<double, 6, 1> force_target; 
  force_target.setZero();
  force_target(2) = -10; // Example: 10N downwards force

  setControl(position, orientation, force_target, q_init_);

  // get state variables
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array = model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  // convert to Eigen
  Eigen::Map<Vector7d> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  Vector7d dq = Eigen::Map<const Vector7d>(robot_state.dq.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d current_position(transform.translation());
  Eigen::Quaterniond current_orientation(transform.rotation());

  // compute motion error
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << current_position - position_d_;
  if (orientation_d_.coeffs().dot(current_orientation.coeffs()) < 0.0) {
    current_orientation.coeffs() << -current_orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(current_orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);

  // compute force error
  Eigen::Map<const Vector7d> tau_measured(robot_state.tau_J.data());
  Eigen::Map<const Vector7d> gravity(model_->gravity(robot_state).data());
  Eigen::MatrixXd j_t = jacobian.transpose();
  Eigen::MatrixXd j_inv = j_t.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::Matrix<double, 6, 1> f_measured = j_inv * (tau_measured - gravity);
  Eigen::Matrix<double, 6, 1> f_error = f_d_ - f_measured;
  force_error_integral_ += duration.toSec() * f_error;

  // Hybrid control law
  Eigen::Matrix<double, 6, 1> command;
  for (int i = 0; i < 6; ++i) {
    if (selection_(i)) { // Force control
      command(i) = f_d_(i) + k_p_ * f_error(i) + k_i_ * force_error_integral_(i);
    } else { // Position control
      command(i) = -K_p_(i, i) * error(i) - K_d_(i, i) * (jacobian * dq)(i);
    }
  }

  // compute control
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  tau_task << jacobian.transpose() * command;
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) - jacobian.transpose() * jacobian_transpose_pinv) * (nullspace_stiffness_ * (q_nullspace_d_ - q) - (2.0 * sqrt(nullspace_stiffness_)) * dq);
  tau_d << tau_task + tau_nullspace + coriolis;

  franka::Torques torques = VectorToArray<7>(tau_d);

  if (getTime() > traj_->getDuration()) {
      bool at_rest = true;
      for (auto dq_i : robot_state.dq) {
          if (std::abs(dq_i) > dq_threshold_) {
              at_rest = false;
          }
      }
      if (at_rest) {
          torques.motion_finished = true;
      }
  }

  return torques;
}

void HybridForceMotion::_updateFilter() {
  K_p_ = ema_filter(K_p_, K_p_target_, filter_coeff_, true);
  K_d_ = ema_filter(K_d_, K_d_target_, filter_coeff_, true);
  nullspace_stiffness_ = ema_filter(nullspace_stiffness_, nullspace_stiffnes_target_, filter_coeff_, true);
  position_d_ = ema_filter(position_d_, position_d_target_, filter_coeff_, true);
  orientation_d_ = orientation_d_.slerp(filter_coeff_, orientation_d_target_);
  f_d_ = ema_filter(f_d_, f_d_target_, filter_coeff_, true);
  k_p_ = ema_filter(k_p_, k_p_target_, filter_coeff_, true);
  k_i_ = ema_filter(k_i_, k_i_target_, filter_coeff_, true);
  selection_ = selection_target_;
}

void HybridForceMotion::setControl(const Eigen::Vector3d& position, const Eigen::Vector4d& orientation, const Eigen::Matrix<double, 6, 1>& force, const Vector7d& q_nullspace) {
  std::lock_guard<std::mutex> lock(mux_);
  position_d_target_ = position;
  orientation_d_target_ = Eigen::Quaterniond(orientation);
  f_d_target_ = force;
  q_nullspace_d_target_ = q_nullspace;
}

void HybridForceMotion::setImpedance(const Eigen::Matrix<double, 6, 6>& impedance) {
  std::lock_guard<std::mutex> lock(mux_);
  K_p_target_ = impedance;
  _computeDamping();
}

void HybridForceMotion::setDampingRatio(const double& damping_ratio) {
  std::lock_guard<std::mutex> lock(mux_);
  damping_ratio_ = damping_ratio;
  _computeDamping();
}

void HybridForceMotion::setNullspaceStiffness(const double& nullspace_stiffness) {
  std::lock_guard<std::mutex> lock(mux_);
  nullspace_stiffnes_target_ = nullspace_stiffness;
}

void HybridForceMotion::setForceProportionalGain(const double& k_p) {
  std::lock_guard<std::mutex> lock(mux_);
  k_p_target_ = k_p;
}

void HybridForceMotion::setForceIntegralGain(const double& k_i) {
  std::lock_guard<std::mutex> lock(mux_);
  k_i_target_ = k_i;
}

void HybridForceMotion::setSelection(const Eigen::Matrix<bool, 6, 1>& selection) {
  std::lock_guard<std::mutex> lock(mux_);
  selection_target_ = selection;
}

void HybridForceMotion::setFilter(const double filter_coeff) {
  std::lock_guard<std::mutex> lock(mux_);
  filter_coeff_ = filter_coeff;
}

void HybridForceMotion::start(const franka::RobotState& robot_state, std::shared_ptr<franka::Model> model) {
  motion_finished_ = false;
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  position_d_ = position;
  position_d_target_ = position;
  orientation_d_ = orientation;
  orientation_d_target_ = orientation;
  q_nullspace_d_ = q;
  q_nullspace_d_target_ = q;
  f_d_.setZero();
  f_d_target_.setZero();
  force_error_integral_.setZero();
  model_ = model;
}

void HybridForceMotion::stop(const franka::RobotState& robot_state, std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool HybridForceMotion::isRunning() { return !motion_finished_; }

const std::string HybridForceMotion::name() { return "Hybrid Force Motion"; }
