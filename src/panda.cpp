#include "panda.h"

#include <franka/exception.h>

#include <iostream>
#include <typeinfo>

#include "constants.h"
#include "kinematics/fk.h"
#include "kinematics/ik.h"
#include "motion/generators.h"

namespace std {
template <typename T, u_long V>
std::ostream &operator<<(std::ostream &os, const std::array<T, V> &vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}  // namespace std

// ---------------------------------------------------------------------------
// Local helpers for robust trajectory construction
// ---------------------------------------------------------------------------

/// Remove consecutive waypoints closer than *threshold* (inf-norm on joints).
/// Always keeps the first and last waypoint.
static std::vector<Vector7d> prefilterWaypoints(
    const std::vector<Vector7d> &waypoints, double threshold) {
  if (waypoints.size() < 2) return waypoints;
  std::vector<Vector7d> filtered;
  filtered.push_back(waypoints.front());
  for (size_t i = 1; i < waypoints.size(); ++i) {
    if ((waypoints[i] - filtered.back()).lpNorm<Eigen::Infinity>() >= threshold) {
      filtered.push_back(waypoints[i]);
    }
  }
  // Ensure the last waypoint is always included
  if (!filtered.back().isApprox(waypoints.back(), 1e-12)) {
    filtered.push_back(waypoints.back());
  }
  return filtered;
}

/// Try building a JointTrajectory with progressively fewer waypoints
/// (full → half → quarter → … down to *min_waypoints*).
/// Returns nullptr on failure.
static std::shared_ptr<motion::JointTrajectory> buildJointTrajectoryRobust(
    const std::vector<Vector7d> &waypoints, double speed_factor,
    double max_deviation, int min_waypoints = 10) {
  // Prefilter near-duplicate waypoints (threshold = 2.5× max_deviation)
  auto filtered = prefilterWaypoints(waypoints, 2.5 * max_deviation);
  int n_full = static_cast<int>(filtered.size());

  // Candidate counts: full, half, quarter, … down to min_waypoints
  std::vector<int> candidates;
  for (int n = n_full; n >= min_waypoints; n /= 2) {
    candidates.push_back(n);
  }
  if (candidates.empty()) {
    candidates.push_back(n_full);
  }

  for (int n_try : candidates) {
    std::vector<Vector7d> wps;
    if (n_try >= n_full) {
      wps = filtered;
    } else {
      // Uniformly subsample
      wps.reserve(n_try);
      for (int i = 0; i < n_try; ++i) {
        int idx = static_cast<int>(
            std::round(static_cast<double>(i) * (n_full - 1) / (n_try - 1)));
        wps.push_back(filtered[idx]);
      }
    }
    try {
      return std::make_shared<motion::JointTrajectory>(
          wps, speed_factor, max_deviation);
    } catch (...) {
      continue;
    }
  }
  return nullptr;  // all attempts failed
}

bool PandaContext::ok() {
  panda_.raiseError();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::high_resolution_clock::now() - t_prev_)
                     .count() /
                 1e6;
  if (elapsed < dt_) {
    std::this_thread::sleep_for(
        std::chrono::microseconds(int((dt_ - elapsed) * 1e6)));
  }
  t_prev_ = std::chrono::high_resolution_clock::now();
  num_ticks_++;
  if (max_ticks_ > 0 && num_ticks_ - 1 >= max_ticks_) {
    return false;
  } else if (t_max_ > 0.0 && getTime() >= t_max_) {
    return false;
  } else {
    return true;
  }
}

PandaContext::PandaContext(Panda &panda, const double &frequency,
                           const double &t_max, const uint64_t &max_ticks)
    : dt_(1.0 / frequency),
      t_prev_(t_start_),
      max_ticks_(max_ticks),
      t_max_(t_max),
      num_ticks_(0),
      panda_(panda) {}

double PandaContext::getTime() {
  return std::chrono::duration_cast<std::chrono::microseconds>(t_prev_ -
                                                               t_start_)
             .count() *
         1e-6;
}

const PandaContext &PandaContext::enter() {
  t_start_ = std::chrono::high_resolution_clock::now();
  return *this;
}

bool PandaContext::exit(const py::object &type, const py::object &value,
                        const py::object &traceback) {
  return false;
}

uint64_t PandaContext::getNumTicks() { return num_ticks_; }

template <typename... Args>
void Panda::_log(const std::string level, Args &&...args) {
  py::gil_scoped_acquire acquire;
  logger_.attr(level.c_str())(args...);
}

Panda::Panda(std::string hostname, std::string name,
             franka::RealtimeConfig realtime_config)
    : name_(name) {
  py::object logging = py::module_::import("logging");
  logger_ = logging.attr("getLogger")(name);
  py::gil_scoped_release release;
  robot_ = std::shared_ptr<franka::Robot>(
      new franka::Robot(hostname, realtime_config));
  model_ = std::make_shared<franka::Model>(robot_->loadModel());
  hostname_ = hostname;
  _log("info", "Connected to robot (%s).", hostname_);
  _setState(robot_->readOnce());
  virtual_walls_ =
      std::shared_ptr<controllers::joint_limits::VirtualWallController>(
          new controllers::joint_limits::VirtualWallController(
              kUpperJointLimits, kLowerJointLimits, kPDZoneWidth, kDZoneWidth,
              kPDZoneStiffness, kPDZoneDamping, kDZoneDamping));
}

Panda::~Panda() {
  _log("info", "Panda class destructor invoked (%s).", hostname_);
  stopController();
}

const PandaContext Panda::createContext(double frequency, double max_runtime,
                                        uint64_t max_iter) {
  return PandaContext(*this, frequency, max_runtime, max_iter);
}

franka::Robot &Panda::getRobot() { return *robot_; }

franka::Model &Panda::getModel() { return *model_; }

franka::RobotState Panda::getState() {
  std::lock_guard<std::mutex> lock(mux_);
  return state_;
}

void Panda::enableLogging(size_t buffer_size) {
  std::lock_guard<std::mutex> lock(mux_);
  log_enabled_ = true;
  log_size_ = buffer_size;
  log_.clear();
}

void Panda::disableLogging() {
  std::lock_guard<std::mutex> lock(mux_);
  log_enabled_ = false;
}

// std::deque<franka::RobotState> Panda::getLog() { return log_; }

std::map<std::string, std::list<Eigen::VectorXd>> Panda::getLog() {
  std::map<std::string, std::list<Eigen::VectorXd>> log;
  std::list<Eigen::VectorXd> O_T_EE, elbow, tau_J, control_command_success_rate,
      O_F_ext_hat_K, K_F_ext_hat_K, q, dq, tau_ext_hat_filtered, time,
      F_T_EE, EE_T_K, m_total, F_x_Ctotal, I_total;
  std::lock_guard<std::mutex> lock(mux_);
  for (auto l : log_) {
    O_T_EE.push_back(Eigen::Map<Eigen::VectorXd>(l.O_T_EE.data(), 16, 1));
    elbow.push_back(Eigen::Map<Eigen::VectorXd>(l.elbow.data(), 2, 1));
    tau_J.push_back(Eigen::Map<Eigen::VectorXd>(l.tau_J.data(), 7, 1));
    control_command_success_rate.push_back(
        Eigen::Matrix<double, 1, 1>::Constant(l.control_command_success_rate));
    O_F_ext_hat_K.push_back(
        Eigen::Map<Eigen::VectorXd>(l.O_F_ext_hat_K.data(), 6, 1));
    K_F_ext_hat_K.push_back(
        Eigen::Map<Eigen::VectorXd>(l.K_F_ext_hat_K.data(), 6, 1));
    q.push_back(Eigen::Map<Eigen::VectorXd>(l.q.data(), 7, 1));
    dq.push_back(Eigen::Map<Eigen::VectorXd>(l.dq.data(), 7, 1));
    tau_ext_hat_filtered.push_back(
        Eigen::Map<Eigen::VectorXd>(l.tau_ext_hat_filtered.data(), 7, 1));
    time.push_back(Eigen::Matrix<double, 1, 1>::Constant(l.time.toMSec()));
    F_T_EE.push_back(Eigen::Map<Eigen::VectorXd>(l.F_T_EE.data(), 16, 1));
    EE_T_K.push_back(Eigen::Map<Eigen::VectorXd>(l.EE_T_K.data(), 16, 1));
    m_total.push_back(Eigen::Matrix<double, 1, 1>::Constant(l.m_total));
    F_x_Ctotal.push_back(Eigen::Map<Eigen::VectorXd>(l.F_x_Ctotal.data(), 3, 1));
    I_total.push_back(Eigen::Map<Eigen::VectorXd>(l.I_total.data(), 9, 1));
  }
  log.emplace("O_T_EE", O_T_EE);
  log.emplace("elbow", elbow);
  log.emplace("tau_J", tau_J);
  log.emplace("control_command_success_rate", control_command_success_rate);
  log.emplace("O_F_ext_hat_K", O_F_ext_hat_K);
  log.emplace("K_F_ext_hat_K", K_F_ext_hat_K);
  log.emplace("q", q);
  log.emplace("dq", dq);
  log.emplace("tau_ext_hat_filtered", tau_ext_hat_filtered);
  log.emplace("time", time);
  log.emplace("F_T_EE", F_T_EE);
  log.emplace("EE_T_K", EE_T_K);
  log.emplace("m_total", m_total);
  log.emplace("F_x_Ctotal", F_x_Ctotal);
  log.emplace("I_total", I_total);

  return log;
}

Eigen::Vector3d Panda::getPosition() {
  std::lock_guard<std::mutex> lock(mux_);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(state_.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  return position;
}

Eigen::Vector4d Panda::getOrientation(bool scalar_first) {
  if (scalar_first) {
    return getOrientationScalarFirst();
  }
  return getOrientationScalarLast();
}

Eigen::Vector4d Panda::getOrientationScalarLast() {
  std::lock_guard<std::mutex> lock(mux_);
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(state_.O_T_EE.data()));
  Eigen::Quaterniond orientation(transform.rotation());
  orientation.normalize();
  return orientation.coeffs();
}

Eigen::Vector4d Panda::getOrientationScalarFirst() {
  Eigen::Vector4d orientation = getOrientationScalarLast();
  Eigen::Vector4d tmp;
  tmp[0] = orientation[3];
  tmp.tail(3) << orientation.head(3);
  return tmp;
}

Vector7d Panda::getJointPositions() {
  std::lock_guard<std::mutex> lock(mux_);
  return Eigen::Map<Vector7d>(state_.q.data());
}

Eigen::Matrix4d Panda::getPose() {
  std::lock_guard<std::mutex> lock(mux_);
  return Eigen::Matrix4d::Map(state_.O_T_EE.data());
}

void Panda::_setState(const franka::RobotState &state) {
  std::lock_guard<std::mutex> lock(mux_);
  state_ = state;
  mux_.unlock();
  if (log_enabled_) {
    log_.push_back(state);
    if (log_.size() > log_size_) {
      log_.pop_front();
    }
  }
}

void Panda::startController(std::shared_ptr<TorqueController> controller_ptr) {
  stopController();
  _startController(controller_ptr);
  current_thread_ = std::thread(
      std::bind(&Panda::_runController, this, _createTorqueCallback()));
}

void Panda::_startController(std::shared_ptr<TorqueController> controller_ptr) {
  recover();
  _log("info", "Starting new controller (%s).", controller_ptr->name());
  virtual_walls_->reset();
  this->current_controller_ = controller_ptr;
  current_controller_->setTime(0);
  auto start_state = robot_->readOnce();
  current_controller_->start(start_state, model_);
}

TorqueCallback Panda::_createTorqueCallback() {
  return TorqueCallback([&](const franka::RobotState &robot_state,
                            franka::Duration duration) -> franka::Torques {
    _setState(robot_state);
    franka::Torques tau = franka::Torques({0, 0, 0, 0, 0, 0, 0});
    if (current_controller_) {
      current_controller_->setTime(current_controller_->getTime() +
                                   duration.toSec());
      tau = current_controller_->step(robot_state, duration);
    }
    // Virtual joint walls
    Array7d tau_virtual_wall, tau_saturated, tau_clipped;
    virtual_walls_->computeTorque(robot_state.q, robot_state.dq,
                                  tau_virtual_wall);
    for (int i = 0; i < 7; i++) {
      tau.tau_J[i] += tau_virtual_wall[i];
    }
    tau_saturated = saturateTorqueRate(tau.tau_J, robot_state.tau_J_d);
    tau_clipped = clipTorques(tau_saturated);
    tau.tau_J = tau_clipped;
    return tau;
  });
}

void Panda::stopController() {
  if (current_controller_ /*&& current_controller_->isRunning()*/) {
    _log("info", "Stopping active controller (%s).",
         current_controller_->name());
    current_controller_->stop(state_, model_);
  }
  if (current_thread_.joinable()) {
    current_thread_.join();
  }
}

void Panda::recover() {
  auto state = robot_->readOnce();
  if (state.current_errors || state.robot_mode == franka::RobotMode::kReflex ||
      state.robot_mode == franka::RobotMode::kOther) {
    _log("warning",
         "Irregular state detected. Attempting automatic error recovery.");
    robot_->automaticErrorRecovery();
  }
}

void Panda::_runController(TorqueCallback &control_callback) {
  try {
    robot_->control(control_callback);
  } catch (const franka::Exception &e) {
    _log("error", "Control loop interruped: %s", e.what());
    last_error_ = std::make_shared<franka::Exception>(e);
  }
}

void Panda::raiseError() {
  if (last_error_) {
    franka::Exception e = *last_error_;
    last_error_.reset();
    throw e;
  }
}

const double Panda::kMoveToJointPositionThreshold = 1e-2;

bool Panda::moveToJointPosition(const Vector7d &position, double speed_factor,
                                const Vector7d &stiffness,
                                const Vector7d &damping, double dq_threshold,
                                double success_threshold) {
  std::vector<Vector7d> waypoints;
  waypoints.push_back(position);
  return moveToJointPosition(waypoints, speed_factor, stiffness, damping,
                             dq_threshold, success_threshold);
}

const double kDefaultTeachingDampingData[7] = {0, 0, 0, 0, 0, 0, 0};
const Vector7d Panda::kDefaultTeachingDamping =
    Vector7d(kDefaultTeachingDampingData);

void Panda::teaching_mode(bool active, const Vector7d &damping) {
  stopController();
  recover();
  if (!active) {
    return;
  }
  auto ctrl = std::make_shared<AppliedTorque>(damping, 1.0);
  startController(ctrl);
}

bool Panda::moveToJointPosition(std::vector<Vector7d> &waypoints,
                                double speed_factor, const Vector7d &stiffness,
                                const Vector7d &damping, double dq_threshold,
                                double success_threshold) {
  stopController();
  recover();
  _setState(robot_->readOnce());
  _log("info", "Initializing motion generation (moveToJointPosition).");
  waypoints.push_back(getJointPositions());
  std::rotate(waypoints.rbegin(), waypoints.rbegin() + 1, waypoints.rend());
  auto traj =
      std::make_shared<motion::JointTrajectory>(waypoints, speed_factor, 0.002);
  if (traj->getDuration() == 0.0) {
    _log("info", "Already at goal.");
    return true;
  }
  auto ctrl = std::make_shared<controllers::JointTrajectory>(
      traj, stiffness, damping, dq_threshold);
  _startController(ctrl);
  auto cb = _createTorqueCallback();
  _runController(cb);
  const Vector7d q = Eigen::Map<const Vector7d>(robot_->readOnce().q.data());
  return waypoints.back().isApprox(q, success_threshold);
}

bool Panda::moveToJointPositionWithHeightLimit(
    const Vector7d &position, double height_limit, double speed_factor,
    double dt, double max_deviation,
    const Vector7d &stiffness, const Vector7d &damping, double dq_threshold,
    double success_threshold) {
  std::vector<Vector7d> waypoints;
  waypoints.push_back(position);
  return moveToJointPositionWithHeightLimit(waypoints, height_limit,
                                           speed_factor, dt, max_deviation,
                                           stiffness, damping,
                                           dq_threshold, success_threshold);
}

bool Panda::moveToJointPositionWithHeightLimit(
    std::vector<Vector7d> &waypoints, double height_limit, double speed_factor,
    double dt, double max_deviation,
    const Vector7d &stiffness, const Vector7d &damping, double dq_threshold,
    double success_threshold) {
  stopController();
  recover();
  _setState(robot_->readOnce());
  _log("info",
       "Initializing motion generation "
       "(moveToJointPositionWithHeightLimit, z_min=%.4f, dt=%.4f, max_dev=%.6f).",
       height_limit, dt, max_deviation);

  // Insert current position as the first waypoint
  waypoints.push_back(getJointPositions());
  std::rotate(waypoints.rbegin(), waypoints.rbegin() + 1, waypoints.rend());

  // Validate that start and goal positions don't violate the limit
  double z_start = kinematics::fk(waypoints.front())(2, 3);
  double z_goal = kinematics::fk(waypoints.back())(2, 3);
  if (z_start < height_limit) {
    _log("error",
         "Current position already violates height limit (z=%.4f < %.4f).",
         z_start, height_limit);
    return false;
  }
  if (z_goal < height_limit) {
    _log("error",
         "Target position violates height limit (z=%.4f < %.4f).",
         z_goal, height_limit);
    return false;
  }

  // Build base trajectory with prefiltering + gradual reduction
  auto base_traj = buildJointTrajectoryRobust(
      waypoints, speed_factor, motion::kDefaultBaseMaxDeviation);
  if (!base_traj) {
    _log("error", "Failed to build base JointTrajectory (all waypoint counts failed).");
    return false;
  }
  if (base_traj->getDuration() == 0.0) {
    _log("info", "Already at goal.");
    return true;
  }

  // Check if the trajectory violates the height limit
  double duration = base_traj->getDuration();
  double check_dt = duration / 200;
  bool has_violation = false;
  for (int i = 0; i <= 200; i++) {
    double t = std::min(i * check_dt, duration);
    Vector7d q_sample = base_traj->getJointPositions(t);
    double z = kinematics::fk(q_sample)(2, 3);
    if (z < height_limit) {
      has_violation = true;
      _log("warning", "Height violation detected at t=%.4f s (z=%.4f < %.4f).",
           t, z, height_limit);
      break;
    }
  }

  std::shared_ptr<motion::JointTrajectory> traj;
  if (!has_violation) {
    _log("info", "Trajectory is height-safe. Executing directly.");
    traj = base_traj;
  } else {
    _log("warning",
         "Height violation detected. Computing corrected trajectory.");
    try {
      traj = std::make_shared<motion::HeightConstrainedJointTrajectory>(
          base_traj, height_limit, dt, max_deviation, speed_factor);
    } catch (const std::exception &e) {
      _log("error",
           "Height-constrained trajectory construction failed: %s. "
           "Refusing to execute.", e.what());
      return false;
    }
  }

  // ── Final safety gate ────────────────────────────────────────────────────
  // Densely re-sample the trajectory that is about to be executed and hard-
  // refuse if ANY sample is below the height limit. This is independent of
  // the internal verification inside HeightConstrainedJointTrajectory and
  // also covers the has_violation==false path (base trajectory).
  {
    double traj_dur = traj->getDuration();
    const int kSafetySteps = 1000;
    double safety_dt = traj_dur / kSafetySteps;
    double z_min_traj = std::numeric_limits<double>::max();
    double z_min_traj_t = 0.0;
    bool safety_violated = false;
    double worst_viol = 0.0;
    double worst_viol_t = 0.0;
    for (int i = 0; i <= kSafetySteps; i++) {
      double t = std::min(i * safety_dt, traj_dur);
      Vector7d q_s = traj->getJointPositions(t);
      double z = kinematics::fk(q_s)(2, 3);
      if (z < z_min_traj) { z_min_traj = z; z_min_traj_t = t; }
      if (z < height_limit) {
        safety_violated = true;
        double v = height_limit - z;
        if (v > worst_viol) { worst_viol = v; worst_viol_t = t; }
      }
    }
    _log("info",
         "Safety gate: z_min=%.4f at t=%.2fs (limit=%.4f, duration=%.2fs).",
         z_min_traj, z_min_traj_t, height_limit, traj_dur);
    if (safety_violated) {
      _log("error",
           "SAFETY GATE BLOCKED execution: trajectory goes %.4f m below "
           "height limit (%.4f) at t=%.2fs. "
           "Reduce dt for denser height correction and retry.",
           worst_viol, height_limit, worst_viol_t);
      return false;
    }
    _log("info", "Safety gate passed (z_min=%.4f >= limit=%.4f). Executing.",
         z_min_traj, height_limit);
  }

  return executeTrajectory(traj, stiffness, damping, dq_threshold,
                           success_threshold);
}

std::shared_ptr<motion::JointTrajectory>
Panda::computeTrajectoryWithHeightLimit(
    const Vector7d &position, double height_limit, double speed_factor,
    double dt, double max_deviation) {
  std::vector<Vector7d> waypoints;
  waypoints.push_back(position);
  return computeTrajectoryWithHeightLimit(waypoints, height_limit,
                                          speed_factor, dt, max_deviation);
}

std::shared_ptr<motion::JointTrajectory>
Panda::computeTrajectoryWithHeightLimit(
    std::vector<Vector7d> &waypoints, double height_limit,
    double speed_factor, double dt, double max_deviation) {
  _setState(robot_->readOnce());
  _log("info",
       "Computing trajectory with height limit (z_min=%.4f, dt=%.4f, max_dev=%.6f).",
       height_limit, dt, max_deviation);

  // Insert current position as the first waypoint
  waypoints.push_back(getJointPositions());
  std::rotate(waypoints.rbegin(), waypoints.rbegin() + 1, waypoints.rend());

  // Validate start and goal
  double z_start = kinematics::fk(waypoints.front())(2, 3);
  double z_goal = kinematics::fk(waypoints.back())(2, 3);
  if (z_start < height_limit) {
    throw std::runtime_error(
        "Current position already violates height limit (z=" +
        std::to_string(z_start) + " < " + std::to_string(height_limit) + ").");
  }
  if (z_goal < height_limit) {
    throw std::runtime_error(
        "Target position violates height limit (z=" +
        std::to_string(z_goal) + " < " + std::to_string(height_limit) + ").");
  }

  // Build base trajectory with prefiltering + gradual reduction
  auto base_traj = buildJointTrajectoryRobust(
      waypoints, speed_factor, motion::kDefaultBaseMaxDeviation);
  if (!base_traj) {
    throw std::runtime_error(
        "Failed to build base JointTrajectory (all waypoint counts failed).");
  }
  if (base_traj->getDuration() == 0.0) {
    _log("info", "Already at goal.");
    return base_traj;
  }

  // Check if the trajectory violates the height limit
  double duration = base_traj->getDuration();
  double check_dt = duration / 200;
  bool has_violation = false;
  for (int i = 0; i <= 200; i++) {
    double t = std::min(i * check_dt, duration);
    Vector7d q_sample = base_traj->getJointPositions(t);
    double z = kinematics::fk(q_sample)(2, 3);
    if (z < height_limit) {
      has_violation = true;
      break;
    }
  }

  if (!has_violation) {
    _log("info", "Trajectory is height-safe.");
    return base_traj;
  }

  _log("warning", "Height violation detected. Computing corrected trajectory.");
  return std::make_shared<motion::HeightConstrainedJointTrajectory>(
      base_traj, height_limit, dt, max_deviation, speed_factor);
}

bool Panda::executeTrajectory(
    std::shared_ptr<motion::JointTrajectory> trajectory,
    const Vector7d &stiffness, const Vector7d &damping,
    double dq_threshold, double success_threshold) {
  stopController();
  recover();
  _setState(robot_->readOnce());
  _log("info", "Executing pre-computed trajectory (duration=%.2f s).",
       trajectory->getDuration());
  if (trajectory->getDuration() == 0.0) {
    _log("info", "Trajectory has zero duration. Already at goal.");
    return true;
  }

  // Move to trajectory start if the robot has drifted since compute time.
  // Without this, the PD controller (K_p=600) would produce a torque spike
  // from the position mismatch, triggering power_limit_violation.
  Vector7d q_now = getJointPositions();
  Vector7d q_start = trajectory->getJointPositions(0.0);
  double start_err = (q_now - q_start).norm();
  constexpr double kStartTolerance = 1e-3;  // ~0.06° per joint
  if (start_err > kStartTolerance) {
    _log("info",
         "Robot position differs from trajectory start (err=%.4f rad). "
         "Moving to start position first.",
         start_err);
    std::vector<Vector7d> approach_wps;
    approach_wps.push_back(q_start);
    if (!moveToJointPosition(approach_wps, 0.2, stiffness, damping,
                             dq_threshold, success_threshold)) {
      _log("error", "Failed to move to trajectory start position.");
      return false;
    }
    // Re-read state after approach move
    _setState(robot_->readOnce());
  }

  auto ctrl = std::make_shared<controllers::JointTrajectory>(
      trajectory, stiffness, damping, dq_threshold);
  _startController(ctrl);
  auto cb = _createTorqueCallback();
  _runController(cb);
  _log("info", "executeTrajectory: control loop exited.");
  const Vector7d q =
      Eigen::Map<const Vector7d>(robot_->readOnce().q.data());
  Vector7d q_goal = trajectory->getJointPositions(trajectory->getDuration());
  return q_goal.isApprox(q, success_threshold);
}

bool Panda::moveToPose(const Eigen::Vector3d &position,
                       const Eigen::Matrix<double, 4, 1> &orientation,
                       double speed_factor,
                       const Eigen::Matrix<double, 6, 6> &impedance,
                       const double &damping_ratio,
                       const double &nullspace_stiffness, double dq_threshold,
                       double success_threshold) {
  std::vector<Eigen::Vector3d> positions;
  positions.push_back(position);
  std::vector<Eigen::Matrix<double, 4, 1>> orientations;
  orientations.push_back(orientation);
  return moveToPose(positions, orientations, speed_factor, impedance, damping_ratio,
                    nullspace_stiffness, dq_threshold, success_threshold);
}

bool Panda::moveToPose(std::vector<Eigen::Vector3d> &positions,
                       std::vector<Eigen::Matrix<double, 4, 1>> &orientations,
                       double speed_factor,
                       const Eigen::Matrix<double, 6, 6> &impedance,
                       const double &damping_ratio,
                       const double &nullspace_stiffness, double dq_threshold,
                       double success_threshold) {
  stopController();
  recover();
  _setState(robot_->readOnce());
  _log("info", "Initializing motion generation (moveToPose).");
  positions.push_back(getPosition());
  orientations.push_back(getOrientation());
  std::rotate(positions.rbegin(), positions.rbegin() + 1, positions.rend());
  std::rotate(orientations.rbegin(), orientations.rbegin() + 1,
              orientations.rend());
  auto traj = std::make_shared<motion::CartesianTrajectory>(
      positions, orientations, speed_factor);
  if (traj->getDuration() == 0.0) {
    _log("info", "Already at goal.");
    return true;
  }
  auto ctrl = std::make_shared<controllers::CartesianTrajectory>(
      traj, getJointPositions(), impedance, damping_ratio, nullspace_stiffness, dq_threshold, 1.0);
  _startController(ctrl);
  auto cb = _createTorqueCallback();
  _runController(cb);
  Eigen::Affine3d transform(
      Eigen::Matrix4d::Map(robot_->readOnce().O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  return positions.back().isApprox(position, success_threshold) &&
         orientations.back().isApprox(orientation.coeffs(), success_threshold);
}

bool Panda::moveToPose(const std::vector<Eigen::Matrix<double, 4, 4>> &poses,
                       double speed_factor,
                       const Eigen::Matrix<double, 6, 6> &impedance,
                       const double &damping_ratio,
                       const double &nullspace_stiffness, double dq_threshold,
                       double success_threshold) {
  std::vector<Eigen::Vector3d> positions;
  std::vector<Eigen::Matrix<double, 4, 1>> orientations;
  for (auto p : poses) {
    positions.push_back(MatrixToPosition(p));
    orientations.push_back(MatrixToOrientation(p));
  }
  return moveToPose(positions, orientations, speed_factor, impedance, damping_ratio,
                    nullspace_stiffness, dq_threshold, success_threshold);
}

bool Panda::moveToPose(const Eigen::Matrix<double, 4, 4> &pose,
                       double speed_factor,
                       const Eigen::Matrix<double, 6, 6> &impedance,
                       const double &damping_ratio,
                       const double &nullspace_stiffness, double dq_threshold,
                       double success_threshold) {
  std::vector<Eigen::Matrix<double, 4, 4>> poses;
  poses.push_back(pose);
  return moveToPose(poses, speed_factor, impedance, damping_ratio,
                    nullspace_stiffness, dq_threshold, success_threshold);
}

bool Panda::moveToStart(double speed_factor, const Vector7d &stiffness,
                        const Vector7d &damping, double dq_threshold,
                        double success_threshold) {
  return moveToJointPosition(kJointPositionStart, speed_factor, stiffness,
                             damping, dq_threshold, success_threshold);
}

std::vector<Vector7d> Panda::getJointTrajectory(
    const Vector7d &position, double speed_factor, double dt,
    double max_deviation) {
  std::vector<Vector7d> waypoints;
  waypoints.push_back(position);
  return getJointTrajectory(waypoints, speed_factor, dt, max_deviation);
}

std::vector<Vector7d> Panda::getJointTrajectory(
    std::vector<Vector7d> &waypoints, double speed_factor, double dt,
    double max_deviation) {
  _setState(robot_->readOnce());
  // waypoints.push_back(getJointPositions());
  // std::rotate(waypoints.rbegin(), waypoints.rbegin() + 1, waypoints.rend());
  auto traj = std::make_shared<motion::JointTrajectory>(waypoints, speed_factor, max_deviation);
  
  // Sample the trajectory at dt intervals
  std::vector<Vector7d> positions;
  double duration = traj->getDuration();
  for (double t = 0.0; t <= duration; t += dt) {
    positions.push_back(traj->getJointPositions(t));
  }
  // Always include the final position
  if (positions.empty() || (duration - (positions.size() - 1) * dt) > 1e-6) {
    positions.push_back(traj->getJointPositions(duration));
  }
  return positions;
}

void Panda::update_robot_state() {
  std::lock_guard<std::mutex> lock(mux_);
  state_ = robot_->readOnce();
}

void Panda::setDefaultBehavior() {
  recover();
  _log("info", "Resetting impedance and collision behavior.");
  robot_->setCollisionBehavior({{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                               {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                               {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
                               {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
                               {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                               {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
                               {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
                               {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
  robot_->setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
  robot_->setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
}

