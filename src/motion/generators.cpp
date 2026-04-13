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
  // NOTE: The Trajectory constructor is fully deterministic — same inputs
  // always produce the same result. Retrying with identical parameters is
  // pointless and just wastes time until the timeout.  We try once and
  // report success or failure immediately.
  traj_ = std::make_shared<time_optimal::Trajectory>(path, max_velocity,
                                                     max_acceleration, 1e-3);
  if (traj_->isValid() && !isnan(traj_->getDuration())) {
    return true;
  }
  _log("error",
       "Trajectory computation failed (invalid trajectory). "
       "The time-optimal planner could not find a feasible solution. "
       "Try reducing the number of waypoints or increasing max_deviation.");
  return false;
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
    double dt,
    double max_deviation,
    double speed_factor,
    double timeout,
    int max_waypoints) {
  py::gil_scoped_acquire acquire;
  py::object logging = py::module_::import("logging");
  logger_ = logging.attr("getLogger")("motion");
  py::gil_scoped_release release;

  double base_duration = base->getDuration();
  int num_samples = static_cast<int>(std::ceil(base_duration / dt)) + 1;

  // Adjust dt so we cover the duration exactly
  if (num_samples > 1) {
    dt = base_duration / (num_samples - 1);
  }

  // Compute minimum z from the trajectory (with small margin for safety)
  double z_min_trajectory = std::numeric_limits<double>::max();
  for (int i = 0; i < num_samples; i++) {
    double t = i * dt;
    if (t > base_duration) t = base_duration;
    Vector7d q_sample = base->getJointPositions(t);
    Eigen::Matrix4d pose = kinematics::fk(q_sample);
    double z = pose(2, 3);
    if (z < z_min_trajectory) {
      z_min_trajectory = z;
    }
  }
  
  // The effective height constraint: max of user-specified limit and trajectory min minus margin
  double z_constraint = std::max(height_limit, z_min_trajectory - 0.001);
  
  _log("info",
       "Height constraint: user limit=%.4f, trajectory min=%.4f, effective=%.4f",
       height_limit, z_min_trajectory, z_constraint);

  // Phase 1: Sample the original trajectory and correct violating samples using HQP IK.
  // Use sequential IK seeding: each corrected sample is seeded from the
  // previous (corrected) sample, ensuring joint-space continuity.
  std::vector<Vector7d> corrected_waypoints;
  corrected_waypoints.reserve(num_samples);
  
  Vector7d q_prev = base->getJointPositions(0.0);
  corrected_waypoints.push_back(q_prev);
  int num_corrected = 0;

  for (int i = 1; i < num_samples; i++) {
    double t = i * dt;
    if (t > base_duration) t = base_duration;
    
    Vector7d q_orig = base->getJointPositions(t);
    Eigen::Matrix4d pose = kinematics::fk(q_orig);
    double z = pose(2, 3);

    Vector7d q_sample;
    if (z < height_limit) {
      // Lift the target pose to the height limit
      Eigen::Matrix4d target_pose = pose;
      target_pose(2, 3) = height_limit;

      // Analytical IK methods
      Vector7d q_corrected = kinematics::ik(target_pose, q_prev, q_prev[6]);

      if (std::isnan(q_corrected[0])) {
        // Fallback: default q7
        q_corrected = kinematics::ik(target_pose, q_prev);
      }
      if (std::isnan(q_corrected[0])) {
        // Try all 4 IK solutions and pick nearest to previous
        Eigen::Matrix<double, 4, 7> q_all =
            kinematics::ik_full(target_pose, q_prev, q_prev[6]);
        double best_dist = std::numeric_limits<double>::max();
        for (int row = 0; row < 4; row++) {
          Vector7d candidate = q_all.row(row);
          if (!std::isnan(candidate[0])) {
            double d = (candidate - q_prev).norm();
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
        q_sample = q_orig;
      } else {
        q_sample = q_corrected;
        num_corrected++;
      }
    } else {
      q_sample = q_orig;
    }
    
    corrected_waypoints.push_back(q_sample);
    q_prev = q_sample;
  }

  _log("info",
       "Height-constrained trajectory: %d/%d samples corrected.",
       num_corrected, num_samples);

  // Log max joint jump between consecutive samples to detect discontinuities
  double max_jump = 0.0;
  int max_jump_idx = 0;
  for (size_t i = 1; i < corrected_waypoints.size(); i++) {
    double jump = (corrected_waypoints[i] - corrected_waypoints[i-1]).norm();
    if (jump > max_jump) {
      max_jump = jump;
      max_jump_idx = static_cast<int>(i);
    }
  }
  _log("info",
       "Max joint-space jump: %.6f rad at sample %d (dt=%.4fs, implied vel=%.4f rad/s).",
       max_jump, max_jump_idx, dt, max_jump / dt);

  // Phase 2: Downsample if needed to limit waypoints for re-planning.
  // The time-optimal planner is O(n²) so we need to limit waypoints.
  std::vector<Vector7d> waypoints_for_planning;
  
  if (max_waypoints > 0 && static_cast<int>(corrected_waypoints.size()) > max_waypoints) {
    // Adaptive downsampling: keep first, last, and uniformly sampled points
    // Also keep waypoints at transitions (where correction occurred)
    _log("info",
         "Downsampling from %d to ~%d waypoints for re-planning.",
         corrected_waypoints.size(), max_waypoints);
    
    // Calculate step size to get approximately max_waypoints
    double step = static_cast<double>(corrected_waypoints.size() - 1) / (max_waypoints - 1);
    
    waypoints_for_planning.reserve(max_waypoints);
    waypoints_for_planning.push_back(corrected_waypoints[0]);
    
    double accumulated = 0.0;
    for (size_t i = 1; i < corrected_waypoints.size() - 1; i++) {
      accumulated += 1.0;
      if (accumulated >= step) {
        waypoints_for_planning.push_back(corrected_waypoints[i]);
        accumulated -= step;
      }
    }
    
    // Always include the last point
    waypoints_for_planning.push_back(corrected_waypoints.back());
    
    _log("info",
         "Downsampled to %d waypoints.",
         waypoints_for_planning.size());
  } else {
    waypoints_for_planning = corrected_waypoints;
  }

  // Phase 3: Re-plan through waypoints using time-optimal planner.
  // Use a small max_deviation to allow smoothing of micro-discontinuities.
  // This produces smooth, dynamically feasible velocities and accelerations.
  _log("info",
       "Re-planning with %d waypoints.",
       waypoints_for_planning.size());

  // Pre-filter waypoints to ensure minimum spacing > 2*max_deviation.
  // The time-optimal Path constructor creates circular blend arcs between
  // consecutive waypoints. When waypoints are closer than 2*max_deviation,
  // the blend circles overlap, producing degenerate path segments that cause
  // numerical issues (NaN / segfault).
  if (max_deviation > 0.0) {
    double min_spacing = 2.5 * max_deviation;
    std::vector<Vector7d> filtered;
    filtered.reserve(waypoints_for_planning.size());
    filtered.push_back(waypoints_for_planning.front());

    for (size_t i = 1; i < waypoints_for_planning.size(); i++) {
      double dist = (waypoints_for_planning[i] - filtered.back()).norm();
      if (dist >= min_spacing) {
        filtered.push_back(waypoints_for_planning[i]);
      }
    }

    // Always keep the last waypoint
    if ((filtered.back() - waypoints_for_planning.back()).norm() > 1e-10) {
      filtered.push_back(waypoints_for_planning.back());
    }

    if (filtered.size() < 2) {
      // Not enough waypoints after filtering - use max_deviation=0 instead
      _log("warning",
           "Only %d waypoints after spacing filter (min_spacing=%.6f). "
           "Falling back to max_deviation=0.",
           filtered.size(), min_spacing);
      max_deviation = 0.0;
    } else {
      _log("info",
           "Pre-filtered from %d to %d waypoints (min_spacing=%.6f).",
           waypoints_for_planning.size(), filtered.size(), min_spacing);
      waypoints_for_planning = filtered;
    }
  }

  std::list<Eigen::VectorXd> waypoint_list;
  for (const auto& wp : waypoints_for_planning) {
    waypoint_list.push_back(Eigen::Map<const Eigen::VectorXd>(wp.data(), 7, 1));
  }
  
  // Use provided max_deviation to smooth discontinuities from IK corrections
  time_optimal::Path path(waypoint_list, max_deviation);
  
  // Use provided speed factor for velocity/acceleration limits
  Eigen::VectorXd max_vel = speed_factor * kQMaxVelocity;
  Eigen::VectorXd max_acc = speed_factor * kQMaxAcceleration;
  
  if (!_computeTrajectory(path, max_vel, max_acc, timeout)) {
    throw std::runtime_error(
        "Failed to re-plan height-constrained trajectory. "
        "The corrected waypoints may be too close together or timeout exceeded.");
  }
  
  duration_ = traj_->getDuration();
  
  // Log velocity statistics from the re-planned trajectory
  double max_vel_norm = 0.0;
  double max_vel_time = 0.0;
  for (double t = 0.0; t <= duration_; t += 0.01) {
    Eigen::VectorXd vel = traj_->getVelocity(t);
    double vel_norm = vel.norm();
    if (vel_norm > max_vel_norm) {
      max_vel_norm = vel_norm;
      max_vel_time = t;
    }
  }
  _log("info",
       "Re-planned trajectory duration: %.2f seconds, max velocity norm: %.4f rad/s at t=%.2fs.",
       duration_, max_vel_norm, max_vel_time);
}

double HeightConstrainedJointTrajectory::getDuration() {
  return duration_;
}

Vector7d HeightConstrainedJointTrajectory::getJointPositions(double time) {
  if (time < 0.0) time = 0.0;
  if (time > duration_) time = duration_;
  Eigen::VectorXd pos = traj_->getPosition(time);
  return Vector7d(pos.data());
}

Vector7d HeightConstrainedJointTrajectory::getJointVelocities(double time) {
  if (time < 0.0) time = 0.0;
  if (time > duration_) time = duration_;
  Eigen::VectorXd vel = traj_->getVelocity(time);
  return Vector7d(vel.data());
}

Vector7d HeightConstrainedJointTrajectory::getJointAccelerations(double time) {
  if (time < 0.0) time = 0.0;
  if (time > duration_) time = duration_;
  Eigen::VectorXd acc = traj_->getAcceleration(time);
  return Vector7d(acc.data());
}
