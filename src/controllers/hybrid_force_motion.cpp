#include "controllers/hybrid_force_motion.h"

#include <iostream>
#include <cmath>
#include <ctime>

#include "panda.h"

namespace controllers {

const double kDefaultStiffnessData[7] = {1200, 1200, 1200, 1200, 500, 300, 100};//{50, 50, 50, 20, 20, 20, 10};
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

  // std::cout << "Hybrid Force/Motion Control Step with destination:" << q_d_.transpose() << std::endl;
  // get state variables
  std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
  std::array<double, 42> jacobian_array = model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  Eigen::Map<const Vector7d> gravity(model_->gravity(robot_state).data());

  // convert to Eigen
  Eigen::Map<Vector7d> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Vector7d q = Eigen::Map<const Vector7d>(robot_state.q.data());
  Vector7d dq_state = Eigen::Map<const Vector7d>(robot_state.dq.data());

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
  Vector7d dq = q_d - q;
  Vector7d tau_pd = K_p.asDiagonal() * dq+ K_d.asDiagonal() * (dq_d - dq_state);
  
  // add coriolis compensation
  Vector7d tau_d = tau_pd;// + coriolis;
  // bool sign_difference_found = (dq.array() * tau_d.array()).matrix().minCoeff() < 0;
  // if (sign_difference_found)
  // {
  //   std::cout << "had a different sign dq::" << dq.transpose() << "tau_d:" << tau_d.transpose() << std::endl;
  // }

  // force limit
  // Eigen::Matrix<double, 6, 7> j_inv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
  // Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
  // double tolerance = 1e-6;
  // Eigen::VectorXd singular_values_inv = svd.singularValues();
  // for (long i = 0; i < singular_values_inv.size(); ++i) {
  //     if (singular_values_inv(i) > tolerance) {
  //         singular_values_inv(i) = 1.0 / singular_values_inv(i);
  //     } else {
  //         singular_values_inv(i) = 0;
  //     }
  // }
  // Eigen::MatrixXd j_t_inv = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

  // std::cout << "J transpose dimensions: " << jacobian.transpose().rows() << "x" << jacobian.transpose().cols() << std::endl;

  double tolerance = 1e-6;

  // 2. Prepare the inverse singular values matrix (Sigma_dagger)
  Eigen::VectorXd singular_values_inv = svd.singularValues();

  for (long i = 0; i < singular_values_inv.size(); ++i) {
      if (singular_values_inv(i) > tolerance) {
          singular_values_inv(i) = 1.0 / singular_values_inv(i);
      } else {
          singular_values_inv(i) = 0.0;
      }
  }

  // 3. Construct the pseudo-inverse of J.transpose()
  // Result Matrix = V * Sigma_dagger * U.transpose()
  // V is (6 x min), Sigma_dagger is (min x min), U.transpose() is (min x 7)
  Eigen::Matrix<double, 6, 7> j_T_dagger = 
      svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().transpose();

  // std::cout << "Result (J^T_dagger) dimensions: " << j_T_dagger.rows() << "x" << j_T_dagger.cols() << std::endl;

  // std::cout << "Pseudo-Inverse (J_dagger) dimensions: " << j_dagger.rows() << "x" << j_dagger.cols() << std::endl;
  Eigen::Matrix<double, 6, 1> f_final = j_T_dagger * (tau_d);//- gravity - tau_ext_init_
  //  Eigen::Matrix<double, 6, 1> f_final_no_gravity = j_T_dagger * (tau_d );//- gravity - tau_ext_init_
  Eigen::Matrix<double, 6, 1> f_capped = f_final;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(f_d(i)) > 0) { // Only limit if a limit is specified
      f_capped(i) = std::clamp(f_capped(i), -std::abs(f_d(i)), std::abs(f_d(i)));
    }
  }

  // If any component was capped, apply a corrective torque
  if (!f_final.isApprox(f_capped)) {
    tau_d += jacobian.transpose() * (f_capped - f_final);
    
    // For logging, let's see the new force after correction
    std::cout << "Force limit applied. Old force: " << f_final.head(3).transpose() << std::endl;
    f_final = j_T_dagger * tau_d;
    std::cout << "Force limit applied. New force: " << f_final.head(3).transpose() << std::endl;
  }
  // std::cout << "Force: " << f_final.head(3).transpose() << std::endl;
  time_t current_time = std::time(0);
  char* dt = std::ctime(&current_time);
  std::cout << "Time: " << dt  <<std::endl;
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
}

void HybridForceMotion::stop(const franka::RobotState &robot_state,
                         std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool HybridForceMotion::isRunning() { return !motion_finished_; }

const std::string HybridForceMotion::name() { return "Hybrid Force Motion"; }

} // namespace controllers