#include "controllers/force.h"

#include <franka/exception.h>

#include <iostream>

#include "panda.h"

const double kDefaultDampingData[7] = {1.0, 1.0, 1.0, 1.0, 0.33, 0.33, 0.17};
const Vector7d Force::kDefaultDamping = Vector7d(kDefaultDampingData);

const double Force::kDefaultFilterCoeff = 0.001;
const double Force::kDefaultProportionalGain = 1;
const double Force::kDefaultIntegralGain = 2;
const double Force::kDefaultThreshold = 0.01;

Force::Force(const double &k_p, const double &k_i, const Vector7d &damping,
             const double &threshold, const double &filter_coeff)
    : filter_coeff_(filter_coeff),
      k_p_(k_p),
      k_i_(k_i),
      k_p_target_(k_p),
      k_i_target_(k_i),
      K_d_(damping),
      K_d_target_(damping),
      threshold_(threshold),
      threshold_target_(threshold){};

franka::Torques Force::step(const franka::RobotState &robot_state,
                            franka::Duration &duration) {
  // get state variables
  std::array<double, 42> jacobian_array =
      model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  std::array<double, 7> gravity_array = model_->gravity(robot_state);
  Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<const Vector7d> tau_measured(
      robot_state.tau_J.data());
  Eigen::Map<const Vector7d> gravity(gravity_array.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Map<const Vector7d> dq(robot_state.dq.data());

  // These quantities may be modified outside of the control loop
  mux_.lock();
  _updateFilter();
  Eigen::Vector3d f = f_d_;
  double k_p = k_p_;
  double k_i = k_i_;
  double threshold = threshold_;
  Vector7d K_d = K_d_;
  mux_.unlock();

  // Abort condition
  Eigen::Vector3d pos_error = position_init_ - position;
  if (getTime() > 0 && (position - position_init_).norm() > threshold) {
    throw franka::Exception(name() + ": Distance threshold exceeded.");
  }

  Eigen::VectorXd tau_d(7), force_torque_d(6), tau_ext(7),tau_error(7);
  force_torque_d.setZero();
  force_torque_d.head(3) = f;
  tau_ext << tau_measured - gravity - tau_ext_init_;
  tau_d << jacobian.transpose() * force_torque_d;
  tau_error = tau_d - tau_ext;
  tau_error_integral_ += duration.toSec() * (tau_error);
  // std::cout << "tau_error:" << tau_error.transpose() <<std::endl;
  // std::cout << "tau_d:" << tau_d.transpose() <<std::endl;
  // FF + PI control
  // tau_d << tau_d + k_p * (tau_error) + k_i * tau_error_integral_ - K_d.asDiagonal() * dq;
  
  // std::cout << "tau_ext:" << tau_ext.transpose() << std::endl;
  // std::cout << "tau_measured:" << tau_measured.transpose() << std::endl;
  // std::cout << "tau_ext_init_:" << tau_ext_init_.transpose() << std::endl;
  Eigen::VectorXd f_e(6), force_measured(6),f_e_z(6),pos_e_xy(6);
  Eigen::MatrixXd j_t = jacobian.transpose();
  Eigen::MatrixXd j_inv = j_t.completeOrthogonalDecomposition().pseudoInverse();
  force_measured = j_inv * (tau_measured-gravity- tau_ext_init_);
  f_e_z.setZero();
  f_e_z[2] = force_torque_d[2] - force_measured[2];//probably should save the initial force mesaured
  f_e_z[0] = 1000*pos_error[0];
  f_e_z[1] = 1000*pos_error[1];
  // force_torque_d = force_measured;
  // force_torque_d[2] = f[2];
  
  f_e_integral += duration.toSec() * f_e_z;

  tau_d << jacobian.transpose() * (force_torque_d + k_p * f_e_z) - K_d.asDiagonal() * dq; // + k_i * f_e_integral
  // std::cout << "f_e_z:" << f_e_z.transpose() <<std::endl;
  // std::cout << "force_measured:" << force_measured.transpose() << std::endl;
  // std::cout << "force_torque_d:" << force_torque_d.transpose() << std::endl;

  franka::Torques torques = VectorToArray<7>(tau_d);
  torques.motion_finished = motion_finished_;
  return torques;
}

void Force::_updateFilter() {
  f_d_ = ema_filter(f_d_, f_d_target_, filter_coeff_, true);
  k_p_ = ema_filter(k_p_, k_p_target_, filter_coeff_, true);
  k_i_ = ema_filter(k_i_, k_i_target_, filter_coeff_, true);
  K_d_ = ema_filter(K_d_, K_d_target_, filter_coeff_, true);
  threshold_ = ema_filter(threshold_, threshold_target_, filter_coeff_, true);
}

void Force::setControl(const Eigen::Vector3d &force) {
  std::lock_guard<std::mutex> lock(mux_);
  f_d_target_ = force;
}

void Force::setFilter(const double &filter_coeff) {
  std::lock_guard<std::mutex> lock(mux_);
  filter_coeff_ = filter_coeff;
}

void Force::setProportionalGain(const double &k_p) {
  std::lock_guard<std::mutex> lock(mux_);
  k_p_target_ = k_p;
}

void Force::setIntegralGain(const double &k_i) {
  std::lock_guard<std::mutex> lock(mux_);
  k_i_target_ = k_i;
}

void Force::setThreshold(const double &threshold) {
  std::lock_guard<std::mutex> lock(mux_);
  threshold_target_ = threshold;
}

void Force::setDamping(const Vector7d &damping) {
  std::lock_guard<std::mutex> lock(mux_);
  K_d_target_ = damping;
}

void Force::start(const franka::RobotState &robot_state,
                  std::shared_ptr<franka::Model> model) {
  motion_finished_ = false;
  f_d_.setZero();
  f_d_target_.setZero();

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());

  position_init_ = position;
  model_ = model;

  // Bias torque sensor
  std::array<double, 7> gravity_array = model->gravity(robot_state);
  std::array<double, 7> tau_measured_array = robot_state.tau_J;
  std::array<double, 7> tau_ext_hat_filtered = robot_state.tau_ext_hat_filtered;
  Eigen::Map<Vector7d> initial_tau_measured(
      tau_measured_array.data());
  Eigen::Map<Vector7d> initial_gravity(gravity_array.data());
  Eigen::Map<Vector7d> initial_tau_ext_hat_filtered(
      tau_ext_hat_filtered.data());
  tau_ext_init_ = initial_tau_measured - initial_gravity;
  // tau_ext_init_ = initial_tau_ext_hat_filtered;
  // std::cout << "tau_ext_init_" << tau_ext_init_.transpose() << std::endl;

  // init integrator
  tau_error_integral_.setZero();
  f_e_integral.setZero();
}

void Force::stop(const franka::RobotState &robot_state,
                 std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool Force::isRunning() { return !motion_finished_; }

const std::string Force::name() { return "Force Controller"; }