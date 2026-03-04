#ifndef PARABOLIC_BLEND_SMOOTHER_H_
#define PARABOLIC_BLEND_SMOOTHER_H_

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <list>
#include <memory>
#include <vector>

#include "kinematics/fk.h"
#include "kinematics/ik.h"
#include "motion/time_optimal/trajectory.h"
#include "utils.h"

namespace py = pybind11;
namespace motion {

const double kDefaultTimeout = 30.0;
const double kDefaultJointSpeedFactor = 0.2;
const double kDefaultCartesianSpeedFactor = 0.2;

class PandaTrajectory {
 public:
  virtual double getDuration() { return traj_->getDuration(); }
  virtual ~PandaTrajectory() = default;

 protected:
  bool _computeTrajectory(const time_optimal::Path &path,
                          const Eigen::VectorXd &max_velocity,
                          const Eigen::VectorXd &max_acceleration,
                          double timeout);

  template <typename... Args>
  void _log(const std::string level, Args &&...args) {
    py::gil_scoped_acquire acquire;
    logger_.attr(level.c_str())(args...);
  }

  py::object logger_;
  std::shared_ptr<time_optimal::Trajectory> traj_;
};

class JointTrajectory : public PandaTrajectory {
 public:
  JointTrajectory() = default;
  JointTrajectory(const std::vector<Vector7d> &waypoints,
                  double speed_factor = kDefaultJointSpeedFactor,
                  double maxDeviation = 0.0, double timeout = kDefaultTimeout);

  virtual Vector7d getJointPositions(double time);

  virtual Vector7d getJointVelocities(double time);

  virtual Vector7d getJointAccelerations(double time);

 private:
  time_optimal::Path _convertList(const std::vector<Vector7d> &list,
                                  double maxDeviation = 0.0);
};

/**
 * A joint trajectory that enforces a minimum end-effector height.
 *
 * It wraps an existing JointTrajectory, densely samples it, and for every
 * sample whose FK z-coordinate falls below the height limit, it lifts the
 * Cartesian pose to the limit and solves IK. The corrected joint positions
 * are stored and velocities/accelerations are computed via finite differences.
 * This replaces the trajectory algorithm for violating sections while keeping
 * the original timing.
 */
class HeightConstrainedJointTrajectory : public JointTrajectory {
 public:
  /**
   * @param base        The original (possibly violating) joint trajectory.
   * @param height_limit Minimum allowed EE z-coordinate.
   * @param dt          Sampling interval in seconds (default 1 ms).
   */
  HeightConstrainedJointTrajectory(
      std::shared_ptr<JointTrajectory> base,
      double height_limit,
      double dt = 0.001);

  double getDuration() override;
  Vector7d getJointPositions(double time) override;
  Vector7d getJointVelocities(double time) override;
  Vector7d getJointAccelerations(double time) override;

 private:
  double duration_;
  double dt_;
  int num_samples_;
  std::vector<Vector7d> q_;    // corrected joint positions
  std::vector<Vector7d> dq_;   // joint velocities (finite diff)
  std::vector<Vector7d> ddq_;  // joint accelerations (finite diff)

  int _timeToIndex(double time) const;
  double _indexToTime(int index) const;
  // Linear interpolation between two samples
  Vector7d _lerp(const std::vector<Vector7d> &data, double time) const;
};

class CartesianTrajectory : public PandaTrajectory {
 public:
  CartesianTrajectory(
      const std::vector<Eigen::Matrix<double, 3, 1>> &positions,
      const std::vector<Eigen::Matrix<double, 4, 1>> &orientations,
      double speed_factor = kDefaultCartesianSpeedFactor,
      double maxDeviation = 0.0, double timeout = kDefaultTimeout);

  CartesianTrajectory(const std::vector<Eigen::Matrix<double, 4, 4>> &poses,
                      double speed_factor = kDefaultCartesianSpeedFactor,
                      double maxDeviation = 0.0,
                      double timeout = kDefaultTimeout);

  Eigen::Matrix<double, 4, 4> getPose(double time);

  Eigen::Vector3d getPosition(double time);

  Eigen::Vector4d getOrientation(double time);

 private:
  std::vector<double> angles_;
  std::vector<Eigen::Vector3d> axes_;
  std::vector<Eigen::Quaterniond> orientations_;
};

}  // namespace motion

#endif