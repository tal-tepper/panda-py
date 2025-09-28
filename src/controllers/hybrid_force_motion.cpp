#include "controllers/hybrid_force_motion.h"

#include <iostream>
#include <cmath>

#include "panda.h"

namespace controllers {

const double kDefaultStiffnessData[7] = {600, 600, 600, 600, 250, 150, 50};
const Vector7d HybridForceMotion::kDefaultStiffness =
    Vector7d(kDefaultStiffnessData);

const double kDefaultDqdData[7] = {0, 0, 0, 0, 0, 0, 0};
const Vector7d HybridForceMotion::kDefaultDqd = Vector7d(kDefaultDqdData);

const double kDefaultDampingData[7] = {50, 50, 50, 20, 20, 20, 10};
const Vector7d HybridForceMotion::kDefaultDamping = Vector7d(kDefaultDampingData);
const double HybridForceMotion::kDefaultFilterCoeff = 1.0;
const Eigen::Matrix<double, 6, 1> HybridForceMotion::kDefaultForce = (Eigen::Matrix<double, 6, 1>() << 0,0,0,0,0,0).finished();


HybridForceMotion::HybridForceMotion(const Vector7d &stiffness, const Vector7d &damping,
                             const double filter_coeff) {
  K_p_ = stiffness;
  K_p_target_ = stiffness;
  K_d_ = damping;
  K_d_target_ = damping;
  filter_coeff_ = filter_coeff;
};

franka::Torques HybridForceMotion::step(const franka::RobotState &robot_state,
                                    franka::Duration &duration) {
  // get state variables
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array = model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  Eigen::Map<const Vector7d> gravity(model_->gravity(robot_state).data());

  // convert to Eigen
  Eigen::Map<Vector7d> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  Vector7d dq = Eigen::Map<const Vector7d>(robot_state.dq.data());

  // These quantities may be modified outside of the control loop
  mux_.lock();
  _updateFilter();
  Vector7d K_p = K_p_;
  Vector7d K_d = K_d_;
  Vector7d q_d = q_d_;
  Vector7d dq_d = dq_d_;
  Eigen::Matrix<double, 6, 1> f_d = f_d_;
  mux_.unlock();

  // PD control
  Vector7d tau_pd = K_p.asDiagonal() * (q_d - q) ;//+ K_d.asDiagonal() * (dq_d - dq)
  
  // add coriolis compensation
  Vector7d tau_d = tau_pd + coriolis;

  // force limit
  // Eigen::Matrix<double, 6, 7> j_inv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = 1e-6;
  Eigen::VectorXd singular_values_inv = svd.singularValues();
  for (long i = 0; i < singular_values_inv.size(); ++i) {
      if (singular_values_inv(i) > tolerance) {
          singular_values_inv(i) = 1.0 / singular_values_inv(i);
      } else {
          singular_values_inv(i) = 0;
      }
  }
  Eigen::MatrixXd j_t_inv = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();
  Eigen::Matrix<double, 6, 1> f_final = j_t_inv * tau_d;
  double f_final_norm = f_final.head(3).norm();
  if (f_final_norm > std::abs(f_d[2]))
  {
    std::cout << "Force limit reached, before change: " << f_final[0] << "," << f_final[1] << "," << f_final[2] << "   tau_d" << tau_d.transpose() << std::endl;
    if (f_final_norm > 1e-9) { // Avoid division by zero
      double scaling_factor = std::abs(f_d[2]) / f_final_norm;
      tau_d *= scaling_factor;
      f_final = j_t_inv * tau_d;
      std::cout << "Force limit reached, limiting to: " << f_final[0] << "," << f_final[1] << "," << f_final[2] << "  scaling factor:" << scaling_factor << "   tau_d" << tau_d.transpose() << std::endl;
      std::cout << "j_t_inv dimensions: " << j_t_inv.rows() << "x" << j_t_inv.cols() << std::endl;
      std::cout << "j_t_inv:" << j_t_inv << std::endl;
    }
  }

  franka::Torques torques = VectorToArray(tau_d);
  torques.motion_finished = motion_finished_;
  return torques;
}

void HybridForceMotion::_updateFilter() {
  q_d_ = ema_filter(q_d_, q_d_target_, filter_coeff_, true);
  dq_d_ = ema_filter(dq_d_, dq_d_target_, filter_coeff_, true);
  K_p_ = ema_filter(K_p_, K_p_target_, filter_coeff_, true);
  K_d_ = ema_filter(K_d_, K_d_target_, filter_coeff_, true);
  f_d_ = ema_filter(f_d_, f_d_target_, filter_coeff_, true);
}

void HybridForceMotion::setControl(const Vector7d &position, const Eigen::Matrix<double, 6, 1>& force,
                               const Vector7d &velocity) {
  std::lock_guard<std::mutex> lock(mux_);
  q_d_target_ = position;
  dq_d_target_ = velocity;
  f_d_target_ = force;
  // std::cout << "Set target position: " << position.transpose() << " Set target force: " << force.transpose() << std::endl;
}

void HybridForceMotion::setStiffness(const Vector7d &stiffness) {
  std::lock_guard<std::mutex> lock(mux_);
  K_p_target_ = stiffness;
}

void HybridForceMotion::setDamping(const Vector7d &damping) {
  std::lock_guard<std::mutex> lock(mux_);
  K_d_target_ = damping;
}

void HybridForceMotion::setFilter(const double filter_coeff) {
  std::lock_guard<std::mutex> lock(mux_);
  filter_coeff_ = filter_coeff;
}

void HybridForceMotion::start(const franka::RobotState &robot_state,
                          std::shared_ptr<franka::Model> model) {
  motion_finished_ = false;
  q_d_ = Eigen::Map<const Vector7d>(robot_state.q.data());
  q_d_target_ = Eigen::Map<const Vector7d>(robot_state.q.data());
  dq_d_.setZero();
  dq_d_target_.setZero();
  f_d_.setZero();
  f_d_target_.setZero();
  model_ = model;
}

void HybridForceMotion::stop(const franka::RobotState &robot_state,
                         std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool HybridForceMotion::isRunning() { return !motion_finished_; }

const std::string HybridForceMotion::name() { return "Hybrid Force Motion"; }

} // namespace controllers