#pragma once
#include "controllers/cartesian_impedance.h"

namespace controllers {

class PrimitiveTrajectory : public CartesianImpedance {
 public:
  static const double kDefaultDqThreshold;
  static const double kDefaultNullspaceStiffness;
  static const Eigen::Matrix<double, 6, 6> kDefaultImpedance;

  PrimitiveTrajectory(const Vector7d &q_init,
             const Eigen::Matrix<double, 6, 6> &impedance = kDefaultImpedance,
             const double &damping_ratio = kDefaultDampingRatio,
             const double &nullspace_stiffness = kDefaultNullspaceStiffness,
             const double dq_threshold = kDefaultDqThreshold,
             const double filter_coeff = kDefaultFilterCoeff);

  franka::Torques step(const franka::RobotState &robot_state,
                       franka::Duration &duration) override;

  void setControl(const Eigen::Vector3d &position,
                  const Eigen::Matrix3d &orientation);

  const std::string name() override;

 private:
  Vector7d q_init_;
  double dq_threshold_;
  bool control_set_;
  Eigen::Vector3d next_position_;
  Eigen::Matrix3d next_orientation_;
  std::mutex control_mutex_;
};

} // namespace

