#include "controllers/primitive_trajectory.h"

using namespace controllers;

const double PrimitiveTrajectory::kDefaultDqThreshold = 1e-3;
const double PrimitiveTrajectory::kDefaultNullspaceStiffness = 15.0;
// clang-format off
double _data_primitive[36] = {800,   0,   0,  0,  0,  0,
                               0, 800,   0,  0,  0,  0,
                               0,   0, 800,  0,  0,  0,
                               0,   0,   0, 40,  0,  0,
                               0,   0,   0,  0, 40,  0,
                               0,   0,   0,  0,  0, 40};
// clang-format on
const Eigen::Matrix<double, 6, 6> PrimitiveTrajectory::kDefaultImpedance =
    Eigen::Matrix<double, 6, 6>(_data_primitive);

PrimitiveTrajectory::PrimitiveTrajectory(const Vector7d &q_init,
             const Eigen::Matrix<double, 6, 6> &impedance,
             const double &damping_ratio,
             const double &nullspace_stiffness,
             const double dq_threshold,
             const double filter_coeff)
    : CartesianImpedance(impedance, damping_ratio, nullspace_stiffness, filter_coeff),
      dq_threshold_(dq_threshold),
      q_init_(q_init),
      control_set_(false) {}

franka::Torques PrimitiveTrajectory::step(const franka::RobotState &robot_state,
                                 franka::Duration &duration) {
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (control_set_) {
      // Convert rotation matrix to quaternion
      Eigen::Quaterniond quat(next_orientation_);
      Eigen::Vector4d orientation_quat(quat.w(), quat.x(), quat.y(), quat.z());
      CartesianImpedance::setControl(next_position_, orientation_quat, q_init_);
      control_set_ = false;
    }
  }
  
  auto torques = CartesianImpedance::step(robot_state, duration);
  return torques;
}

void PrimitiveTrajectory::setControl(const Eigen::Vector3d &position,
                                     const Eigen::Matrix3d &orientation) {
  std::lock_guard<std::mutex> lock(control_mutex_);
  next_position_ = position;
  next_orientation_ = orientation;
  control_set_ = true;
}

const std::string PrimitiveTrajectory::name() {
  return "PrimitiveTrajectory";
}

