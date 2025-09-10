#pragma once

#include <atomic>
#include <mutex>

#include "controllers/controller.h"
#include "motion/generators.h"
#include "utils.h"

namespace controllers {

class HybridForceMotion : public TorqueController {
 public:
  HybridForceMotion(
      std::shared_ptr<motion::CartesianTrajectory> trajectory,
      const Vector7d& q_init,
      const Eigen::Matrix<double, 6, 6>& impedance =
          kDefaultImpedance,
      const double& damping_ratio = kDefaultDampingRatio,
      const double& nullspace_stiffness = kDefaultNullspaceStiffness,
      const double& force_k_p = kDefaultForceProportionalGain,
      const double& force_k_i = kDefaultForceIntegralGain,
      const Eigen::Matrix<bool, 6, 1>& selection = kDefaultSelection,
      const double& filter_coeff = kDefaultFilterCoeff);

  franka::Torques step(const franka::RobotState& robot_state,
                       franka::Duration& duration) override;

  void setControl(const Eigen::Vector3d& position,
                  const Eigen::Vector4d& orientation,
                  const Eigen::Matrix<double, 6, 1>& force,
                  const Vector7d& q_nullspace = kJointPositionStart);

  void setImpedance(const Eigen::Matrix<double, 6, 6>& impedance);
  void setDampingRatio(const double& damping_ratio);
  void setNullspaceStiffness(const double& nullspace_stiffness);
  void setForceProportionalGain(const double& k_p);
  void setForceIntegralGain(const double& k_i);
  void setSelection(const Eigen::Matrix<bool, 6, 1>& selection);
  void setFilter(const double filter_coeff);

  void start(const franka::RobotState& robot_state,
             std::shared_ptr<franka::Model> model) override;
  void stop(const franka::RobotState& robot_state,
            std::shared_ptr<franka::Model> model) override;
  bool isRunning() override;
  const std::string name() override;

  static const Eigen::Matrix<double, 6, 6> kDefaultImpedance;
  static const double kDefaultDampingRatio;
  static const double kDefaultNullspaceStiffness;
  static const double kDefaultForceProportionalGain;
  static const double kDefaultForceIntegralGain;
  static const Eigen::Matrix<bool, 6, 1> kDefaultSelection;
  static const double kDefaultFilterCoeff;
  static const double kDefaultDqThreshold;

 private:
  void _updateFilter();
  void _computeDamping();

  std::shared_ptr<motion::CartesianTrajectory> traj_;
  Vector7d q_init_;
  double dq_threshold_;

  Eigen::Matrix<double, 6, 6> K_p_, K_d_, K_p_target_, K_d_target_;
  Eigen::Vector3d position_d_, position_d_target_;
  Eigen::Quaterniond orientation_d_, orientation_d_target_;
  Vector7d q_nullspace_d_, q_nullspace_d_target_;

  double nullspace_stiffness_, nullspace_stiffnes_target_, damping_ratio_;

  Eigen::Matrix<double, 6, 1> f_d_, f_d_target_;
  double k_p_, k_i_, k_p_target_, k_i_target_;
  Eigen::Matrix<double, 6, 1> force_error_integral_;

  Eigen::Matrix<bool, 6, 1> selection_, selection_target_;

  double filter_coeff_;
  std::mutex mux_;
  std::atomic<bool> motion_finished_;
  std::shared_ptr<franka::Model> model_;
};

}  // namespace controllers
