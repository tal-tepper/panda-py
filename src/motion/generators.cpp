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

  // Helper: solve IK for target_pose seeded from q_seed.
  // Tries q7-seeded, default, then all 4 branches. Returns NaN vector on failure.
  // Crucially, always picks the solution nearest to q_seed to stay on the same
  // kinematic branch and avoid elbow flips.
  auto solveIKNearest = [&](const Eigen::Matrix4d &target_pose,
                            const Vector7d &q_seed) -> Vector7d {
    // Collect all candidate solutions
    std::vector<Vector7d> candidates;
    {
      Vector7d s = kinematics::ik(target_pose, q_seed, q_seed[6]);
      if (!std::isnan(s[0])) candidates.push_back(s);
    }
    {
      Vector7d s = kinematics::ik(target_pose, q_seed);
      if (!std::isnan(s[0])) candidates.push_back(s);
    }
    {
      Eigen::Matrix<double, 4, 7> q_all =
          kinematics::ik_full(target_pose, q_seed, q_seed[6]);
      for (int row = 0; row < 4; row++) {
        Vector7d s = q_all.row(row);
        if (!std::isnan(s[0])) candidates.push_back(s);
      }
    }
    if (candidates.empty())
      return Vector7d::Constant(std::numeric_limits<double>::quiet_NaN());
    // Return the candidate closest to the seed (preserves IK branch)
    Vector7d best = candidates[0];
    double best_dist = (best - q_seed).norm();
    for (size_t k = 1; k < candidates.size(); k++) {
      double d = (candidates[k] - q_seed).norm();
      if (d < best_dist) { best_dist = d; best = candidates[k]; }
    }
    return best;
  };

  // Phase 1: Sample the base trajectory and collect z values.
  std::vector<Vector7d> samples(num_samples);
  std::vector<double>   z_vals(num_samples);
  for (int i = 0; i < num_samples; i++) {
    double t = std::min(i * dt, base_duration);
    samples[i] = base->getJointPositions(t);
    z_vals[i]  = kinematics::fk(samples[i])(2, 3);
  }

  // Phase 2: Correct every violating sample by lifting z to height_limit and
  // re-solving IK, seeded from the BASE trajectory's joint positions at that
  // same sample index — NOT from the previous corrected sample.
  //
  // This is the critical design choice: seeding from samples[i] (the base traj)
  // means each IK call is a tiny perturbation (only Δz ≈ violation depth), so
  // it stays on the exact same kinematic branch as the original motion. There
  // is no accumulated branch drift across samples, and no discontinuity at the
  // entry/exit of the constrained zone (adjacent base samples are already close).
  std::vector<Vector7d> corrected(num_samples);
  int num_corrected = 0;
  int num_ik_failed = 0;
  double z_min_orig = std::numeric_limits<double>::max();
  double z_min_orig_t = 0.0;

  for (int i = 0; i < num_samples; i++) {
    double t = std::min(i * dt, base_duration);
    if (z_vals[i] < z_min_orig) { z_min_orig = z_vals[i]; z_min_orig_t = t; }

    if (z_vals[i] < height_limit) {
      Eigen::Matrix4d lifted_pose = kinematics::fk(samples[i]);
      lifted_pose(2, 3) = height_limit;
      // Seed from samples[i]: the IK change is only Δz ≈ (height_limit - z_vals[i]),
      // so the solution is always on the same branch.
      Vector7d q_corr = solveIKNearest(lifted_pose, samples[i]);
      if (!std::isnan(q_corr[0])) {
        double z_check = kinematics::fk(q_corr)(2, 3);
        if (z_check < height_limit - 1e-4) {
          _log("warning",
               "IK lift at idx=%d: FK check z=%.4f still below limit=%.4f "
               "(violation=%.4f m). Using corrected point anyway.",
               i, z_check, height_limit, height_limit - z_check);
        }
        corrected[i] = q_corr;
        num_corrected++;
      } else {
        _log("warning",
             "IK failed at idx=%d (t=%.3fs, z=%.4f < %.4f). "
             "Keeping original — this sample will violate the height limit.",
             i, t, z_vals[i], height_limit);
        corrected[i] = samples[i];
        num_ik_failed++;
      }
    } else {
      corrected[i] = samples[i];
    }
  }

  _log("info",
       "Height correction: %d/%d samples lifted, %d IK failures. "
       "Base traj z_min=%.4f at t=%.2fs (limit=%.4f).",
       num_corrected, num_samples, num_ik_failed,
       z_min_orig, z_min_orig_t, height_limit);

  // Downsample: the time-optimal planner is O(n²); limit to max_waypoints.
  // Always keep first and last.
  std::vector<Vector7d> waypoints_for_planning;
  if (max_waypoints > 0 && num_samples > max_waypoints) {
    waypoints_for_planning.reserve(max_waypoints);
    for (int k = 0; k < max_waypoints; k++) {
      int idx = static_cast<int>(
          std::round(static_cast<double>(k) * (num_samples - 1) / (max_waypoints - 1)));
      waypoints_for_planning.push_back(corrected[idx]);
    }
    _log("info", "Downsampled corrected waypoints from %d to %d.", num_samples, max_waypoints);
  } else {
    waypoints_for_planning = corrected;
  }

  _log("info",
       "Re-planning with %d corrected waypoints.",
       static_cast<int>(waypoints_for_planning.size()));

  // Phase 3: Re-plan through the small set of waypoints using the
  // time-optimal planner.
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

  // Post-construction verification: densely sample the re-planned trajectory
  // and confirm every point satisfies the height limit. The time-optimal
  // planner blends in joint space and has no knowledge of Cartesian height,
  // so residual violations are possible even with fully corrected waypoints.
  {
    const int kVerifySteps = 1000;
    double verify_dt = duration_ / kVerifySteps;
    double z_min_final = std::numeric_limits<double>::max();
    double z_min_final_t = 0.0;
    int n_violations = 0;
    double worst_violation = 0.0;
    double worst_violation_t = 0.0;
    for (int i = 0; i <= kVerifySteps; i++) {
      double t = std::min(i * verify_dt, duration_);
      Eigen::VectorXd pos = traj_->getPosition(t);
      Vector7d q_check(pos.data());
      double z = kinematics::fk(q_check)(2, 3);
      if (z < z_min_final) { z_min_final = z; z_min_final_t = t; }
      if (z < height_limit) {
        n_violations++;
        double viol = height_limit - z;
        if (viol > worst_violation) { worst_violation = viol; worst_violation_t = t; }
      }
    }
    _log("info",
         "Re-planned trajectory: duration=%.2fs, z_min=%.4f at t=%.2fs "
         "(limit=%.4f, %d/%d samples checked).",
         duration_, z_min_final, z_min_final_t, height_limit,
         kVerifySteps + 1, kVerifySteps + 1);
    if (n_violations > 0) {
      _log("error",
           "Re-planned trajectory has %d/%d samples below height limit! "
           "Worst: z=%.4f (%.4f m below limit=%.4f) at t=%.2fs. "
           "Reduce dt (denser waypoints) or check IK solutions.",
           n_violations, kVerifySteps + 1,
           height_limit - worst_violation, worst_violation, height_limit,
           worst_violation_t);
      throw std::runtime_error(
          "Height-constrained trajectory violates height limit after re-planning "
          "(worst: " + std::to_string(worst_violation) +
          " m below limit at t=" + std::to_string(worst_violation_t) + " s).");
    }
    _log("info", "Height constraint verified OK: z_min=%.4f >= limit=%.4f.",
         z_min_final, height_limit);
  }
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
