#include "motion/generators.h"

#include <chrono>
#include <iostream>
#include <numeric>

#include "constants.h"
#include "kinematics/fk.h"

using namespace std;
using namespace Eigen;
using namespace motion;

bool PandaTrajectory::_computeTrajectory(
    const time_optimal::Path &path, const Eigen::VectorXd &max_velocity,
    const Eigen::VectorXd &max_acceleration, double timeout) {
  auto startTime = std::chrono::high_resolution_clock::now();
  bool success = false;
  int i = 0;
  while (true) {
    i++;
    traj_ = std::make_shared<time_optimal::Trajectory>(path, max_velocity,
                                                       max_acceleration, 1e-3);
    if (traj_->isValid() && !isnan(traj_->getDuration())) {
      success = true;
      break;
    }
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                        currentTime - startTime)
                        .count();
    if (duration >= timeout) {
      _log("error", "Trajectory computation timed out after %d seconds.",
           duration);
      break;
    }
    _log("debug", "Reattempting trajectory computation. Attempt no. %d.", i);
  }
  return success;
}

JointTrajectory::JointTrajectory(const std::vector<Vector7d> &waypoints,
                                 double speed_factor, double maxDeviation,
                                 double timeout) {
  py::gil_scoped_acquire acquire;
  py::object logging = py::module_::import("logging");
  logger_ = logging.attr("getLogger")("motion");
  py::gil_scoped_release release;

  if (!_computeTrajectory(_convertList(waypoints, maxDeviation),
                          speed_factor * kQMaxVelocity,
                          speed_factor * kQMaxAcceleration, timeout)) {
    throw runtime_error("Trajectory generation faild.");
  }

  if (waypoints.size() == 2) {
    _log("info",
         "Computed joint trajectory: 1 waypoint, duration %.2f seconds.",
         traj_->getDuration());
  } else {
    _log("info",
         "Computed joint trajectory: %d waypoints, duration %.2f seconds.",
         waypoints.size() - 1, traj_->getDuration());
  }
}

time_optimal::Path JointTrajectory::_convertList(
    const std::vector<Vector7d> &list, double maxDeviation) {
  std::list<Eigen::VectorXd> new_list;
  for (auto l : list) {
    new_list.push_back(Eigen::Map<Eigen::VectorXd>(l.data(), 7, 1));
  }
  return time_optimal::Path(new_list, maxDeviation);
}

Vector7d JointTrajectory::getJointPositions(double time) {
  return traj_->getPosition(time);
}

Vector7d JointTrajectory::getJointVelocities(double time) {
  return traj_->getVelocity(time);
}

Vector7d JointTrajectory::getJointAccelerations(double time) {
  return traj_->getAcceleration(time);
}

CartesianTrajectory::CartesianTrajectory(
    const std::vector<Eigen::Matrix<double, 4, 4>> &poses, double speed_factor,
    double maxDeviation, double timeout) {
  std::vector<Eigen::Matrix<double, 3, 1>> positions;
  std::vector<Eigen::Matrix<double, 4, 1>> orientations;
  for (auto p : poses) {
    positions.push_back(MatrixToPosition(p));
    orientations.push_back(MatrixToOrientation(p));
  }
  CartesianTrajectory(positions, orientations, speed_factor, maxDeviation,
                      timeout);
}

CartesianTrajectory::CartesianTrajectory(
    const std::vector<Eigen::Matrix<double, 3, 1>> &positions,
    const std::vector<Eigen::Matrix<double, 4, 1>> &orientations,
    double speed_factor, double maxDeviation, double timeout) {
  py::gil_scoped_acquire acquire;
  py::object logging = py::module_::import("logging");
  logger_ = logging.attr("getLogger")("motion");
  py::gil_scoped_release release;
  angles_.push_back(0);

  for (size_t i = 0; i < orientations.size() - 1; i++) {
    Eigen::Quaterniond q1(orientations.at(i)), q2(orientations.at(i + 1));
    Eigen::AngleAxisd aa(q2 * q1.inverse());
    double angle = aa.angle();
    Eigen::Vector3d axis = aa.axis();
    angles_.push_back(angle);
    axes_.push_back(axis);
    orientations_.push_back(q1);
  }
  orientations_.push_back(Eigen::Quaterniond(orientations.back()));
  std::partial_sum(angles_.begin(), angles_.end(), angles_.begin());
  std::list<Eigen::VectorXd> waypoints;

  for (size_t i = 0; i < orientations.size(); i++) {
    Eigen::VectorXd point;
    point.resize(4);
    point.head(3) = positions.at(i);
    point.coeffRef(3) = angles_.at(i);
    waypoints.push_back(point);
  }

  if (!_computeTrajectory(time_optimal::Path(waypoints, maxDeviation),
                          speed_factor * kXMaxVelocity,
                          speed_factor * kXMaxAcceleration, timeout)) {
    throw runtime_error("Trajectory generation failed.");
  }

  if (orientations.size() == 2) {
    _log("info",
         "Computed Cartesian trajectory: 1 waypoint, duration %.2f seconds.",
         traj_->getDuration());
  } else {
    _log("info",
         "Computed Cartesian trajectory: %d waypoints, duration %.2f seconds.",
         orientations.size() - 1, traj_->getDuration());
  }
}

Eigen::Matrix<double, 4, 4> CartesianTrajectory::getPose(double time) {
  auto pose = traj_->getPosition(time);
  size_t idx = traj_->getTrajectorySegmentIndex(time);
  double angle = pose.coeff(3) - angles_.at(idx);
  Eigen::AngleAxisd aa(angle, axes_.at(idx));
  Eigen::Quaterniond o = Eigen::Quaterniond(aa) * orientations_.at(idx);

  Eigen::Affine3d transform;
  transform = o;
  transform.translation() = pose.head(3);

  return transform.matrix();
}

Eigen::Vector3d CartesianTrajectory::getPosition(double time) {
  auto pose = traj_->getPosition(time);
  return pose.head(3);
}

Eigen::Vector4d CartesianTrajectory::getOrientation(double time) {
  auto pose = traj_->getPosition(time);
  size_t idx = traj_->getTrajectorySegmentIndex(time);
  double angle = pose.coeff(3) - angles_.at(idx);
  Eigen::AngleAxisd aa(angle, axes_.at(idx));
  Eigen::Quaterniond o = Eigen::Quaterniond(aa) * orientations_.at(idx);
  return o.coeffs();
}

// ---------------------------------------------------------------------------
// HeightConstrainedJointTrajectory
// ---------------------------------------------------------------------------

HeightConstrainedJointTrajectory::HeightConstrainedJointTrajectory(
    std::shared_ptr<JointTrajectory> base,
    double height_limit,
    double dt)
    : dt_(dt) {
  py::gil_scoped_acquire acquire;
  py::object logging = py::module_::import("logging");
  logger_ = logging.attr("getLogger")("motion");
  py::gil_scoped_release release;

  duration_ = base->getDuration();
  num_samples_ = static_cast<int>(std::ceil(duration_ / dt_)) + 1;

  // Adjust dt_ so we cover the duration exactly
  if (num_samples_ > 1) {
    dt_ = duration_ / (num_samples_ - 1);
  }

  q_.resize(num_samples_);
  dq_.resize(num_samples_);
  ddq_.resize(num_samples_);

  // Phase 1: Sample the original trajectory and correct violating samples.
  // Use sequential IK seeding: each corrected sample is seeded from the
  // previous (corrected) sample, ensuring joint-space continuity.
  q_[0] = base->getJointPositions(0.0);
  int num_corrected = 0;

  for (int i = 1; i < num_samples_; i++) {
    double t = _indexToTime(i);
    Vector7d q_orig = base->getJointPositions(t);
    Eigen::Matrix4d pose = kinematics::fk(q_orig);
    double z = pose(2, 3);

    if (z < height_limit) {
      // Lift the pose to the height limit, keep everything else
      pose(2, 3) = height_limit;

      // Solve IK seeded from previous corrected sample
      Vector7d q_corrected = kinematics::ik(pose, q_[i - 1], q_[i - 1][6]);

      if (std::isnan(q_corrected[0])) {
        // Fallback: default q7
        q_corrected = kinematics::ik(pose, q_[i - 1]);
      }
      if (std::isnan(q_corrected[0])) {
        // Try all 4 IK solutions and pick nearest to previous
        Eigen::Matrix<double, 4, 7> q_all =
            kinematics::ik_full(pose, q_[i - 1], q_[i - 1][6]);
        double best_dist = std::numeric_limits<double>::max();
        for (int row = 0; row < 4; row++) {
          Vector7d candidate = q_all.row(row);
          if (!std::isnan(candidate[0])) {
            double d = (candidate - q_[i - 1]).norm();
            if (d < best_dist) {
              best_dist = d;
              q_corrected = candidate;
            }
          }
        }
      }
      if (std::isnan(q_corrected[0])) {
        // Last resort: keep original (will still violate but won't crash)
        _log("warning",
             "IK failed at t=%.4f (z=%.4f). Keeping original trajectory.",
             t, z);
        q_[i] = q_orig;
      } else {
        q_[i] = q_corrected;
        num_corrected++;
      }
    } else {
      q_[i] = q_orig;
    }
  }

  _log("info",
       "Height-constrained trajectory: %d/%d samples corrected (dt=%.4f s).",
       num_corrected, num_samples_, dt_);

  // Phase 2: Compute velocities via central finite differences.
  // At boundaries use forward/backward differences.
  for (int i = 0; i < num_samples_; i++) {
    if (i == 0) {
      dq_[i] = (q_[1] - q_[0]) / dt_;
    } else if (i == num_samples_ - 1) {
      dq_[i] = (q_[i] - q_[i - 1]) / dt_;
    } else {
      dq_[i] = (q_[i + 1] - q_[i - 1]) / (2.0 * dt_);
    }
  }

  // Phase 3: Compute accelerations via central finite differences on dq.
  for (int i = 0; i < num_samples_; i++) {
    if (i == 0) {
      ddq_[i] = (dq_[1] - dq_[0]) / dt_;
    } else if (i == num_samples_ - 1) {
      ddq_[i] = (dq_[i] - dq_[i - 1]) / dt_;
    } else {
      ddq_[i] = (q_[i + 1] - 2.0 * q_[i] + q_[i - 1]) / (dt_ * dt_);
    }
  }

  // Enforce zero velocity at start and end for safe start/stop
  dq_[0].setZero();
  dq_[num_samples_ - 1].setZero();
  ddq_[0].setZero();
  ddq_[num_samples_ - 1].setZero();
}

double HeightConstrainedJointTrajectory::getDuration() {
  return duration_;
}

int HeightConstrainedJointTrajectory::_timeToIndex(double time) const {
  int idx = static_cast<int>(time / dt_);
  if (idx < 0) idx = 0;
  if (idx >= num_samples_) idx = num_samples_ - 1;
  return idx;
}

double HeightConstrainedJointTrajectory::_indexToTime(int index) const {
  return index * dt_;
}

Vector7d HeightConstrainedJointTrajectory::_lerp(
    const std::vector<Vector7d> &data, double time) const {
  if (time <= 0.0) return data.front();
  if (time >= duration_) return data.back();

  double idx_f = time / dt_;
  int i0 = static_cast<int>(idx_f);
  if (i0 >= num_samples_ - 1) return data.back();

  double alpha = idx_f - i0;
  return (1.0 - alpha) * data[i0] + alpha * data[i0 + 1];
}

Vector7d HeightConstrainedJointTrajectory::getJointPositions(double time) {
  return _lerp(q_, time);
}

Vector7d HeightConstrainedJointTrajectory::getJointVelocities(double time) {
  return _lerp(dq_, time);
}

Vector7d HeightConstrainedJointTrajectory::getJointAccelerations(double time) {
  return _lerp(ddq_, time);
}
