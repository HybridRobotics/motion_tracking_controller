#include "motion_tracking_controller/SmCarryController.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <pluginlib/class_list_macros.hpp>

namespace legged {
namespace {
nlohmann::json readJson(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open " + path);
  return nlohmann::json::parse(stream);
}
vector_t vectorFrom(const nlohmann::json& value, size_t size) {
  if (!value.is_array() || value.size() != size) throw std::runtime_error("Wrong vector width");
  vector_t result(size);
  for (size_t i = 0; i < size; ++i) result[i] = value.at(i).get<double>();
  if (!result.allFinite()) throw std::runtime_error("Non-finite deployment data");
  return result;
}
matrix3_t yawRotation(const matrix3_t& rotation) {
  return Eigen::AngleAxisd(std::atan2(rotation(1, 0), rotation(0, 0)), vector3_t::UnitZ()).toRotationMatrix();
}
}  // namespace

SmCarryReference::SmCarryReference(const std::string& path) {
  const auto data = readJson(path);
  jointNames_ = data.at("joint_names").get<std::vector<std::string>>();
  offsets_ = data.at("offset_frames").get<std::vector<size_t>>();
  if (jointNames_.size() != 29 || offsets_ != std::vector<size_t>{0,5,10,15,20,25,30,35,40,45})
    throw std::runtime_error("SM carry requires the 29-joint, 10-frame contract");
  if (std::abs(data.at("fps").get<double>() - 50.0) > 1e-6)
    throw std::runtime_error("Reference must be 50 Hz");
  const auto count = data.at("joint_pos").size();
  if (!count || data.at("joint_vel").size() != count || data.at("pelvis_quat_wxyz").size() != count)
    throw std::runtime_error("Reference arrays have inconsistent lengths");
  positions_.resize(count, 29); velocities_.resize(count, 29);
  for (size_t t = 0; t < count; ++t) {
    positions_.row(t) = vectorFrom(data.at("joint_pos").at(t), 29).transpose();
    velocities_.row(t) = vectorFrom(data.at("joint_vel").at(t), 29).transpose();
    const auto q = vectorFrom(data.at("pelvis_quat_wxyz").at(t), 4);
    quaternion_t quat(q[0], q[1], q[2], q[3]);
    if (quat.norm() < 1e-6) throw std::runtime_error("Invalid pelvis quaternion");
    rotations_.push_back(quat.normalized().toRotationMatrix());
  }
}

vector_t SmCarryReference::goal(size_t step, const matrix3_t& pelvisRotation,
                              const matrix3_t& referenceAlignment) const {
  vector_t result(640);
  for (size_t i = 0; i < offsets_.size(); ++i) {
    // Saturate before adding the offset; never wrap into another clip.
    const size_t t = std::min(std::min(step, size() - 1) + offsets_[i], size() - 1);
    result.segment(i * 58, 29) = positions_.row(t).transpose();
    result.segment(i * 58 + 29, 29) = velocities_.row(t).transpose();
    const matrix3_t rotation = pelvisRotation.transpose() * referenceAlignment * rotations_[t];
    result.segment<6>(580 + i * 6) << rotation(0,0), rotation(0,1), rotation(1,0), rotation(1,1), rotation(2,0), rotation(2,1);
  }
  return result;
}

SmCarryOnnxPolicy::SmCarryOnnxPolicy(const std::string& path, const std::string& controlPath)
    : OnnxPolicy(path, 1), control_(readJson(controlPath)) {}

void SmCarryOnnxPolicy::parseMetadata() {
  // The supplied graph has JSON-style metadata, not legged_control2 CSV metadata.
  // Read the explicit prepared contract instead of modifying the source ONNX.
  jointNames_ = control_.at("joint_names").get<std::vector<std::string>>();
  jointStiffness_ = vectorFrom(control_.at("kp"), 29);
  jointDamping_ = vectorFrom(control_.at("kd"), 29);
  defaultJointPositions_ = vectorFrom(control_.at("default_joint_position_rad"), 29);
  actionScale_ = vectorFrom(control_.at("action_scale"), 29);
  if ((defaultJointPositions_ - vectorFrom(control_.at("action_offset_rad"), 29)).norm() > 1e-6)
    throw std::runtime_error("This adapter requires action offset == default joint position");
  if (vectorFrom(control_.at("default_joint_velocity_rad_s"), 29).norm() > 1e-10)
    throw std::runtime_error("Nonzero default joint velocities need a separate observation term");
  actionLimit_ = control_.at("raw_action_limit").get<double>();
  if (!(actionLimit_ > 0 && std::isfinite(actionLimit_)) || jointNames_.size() != 29 ||
      (jointStiffness_.array() <= 0).any() || (jointDamping_.array() <= 0).any())
    throw std::runtime_error("Invalid action or PD contract");
  observationNames_ = {"projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions", "command"};
  observationHistoryLengths_ = {1,1,1,1,1,1};
  commandNames_ = {"sm_carry_motion"};
}

void SmCarryOnnxPolicy::checkInputOutput() {
  OnnxPolicy::checkInputOutput();
  if (inputShapes_.size() != 1 || outputShapes_.size() != 1 || getObservationSize() != 733 || getActionSize() != 29)
    throw std::runtime_error("Expected exactly obs[1,733] -> actions[1,29]");
}

void SmCarryOnnxPolicy::reset() {
  step_ = 0;
  lastObservation_ = vector_t::Zero(733);
  OnnxPolicy::reset();
}

vector_t SmCarryOnnxPolicy::forward(const vector_t& observations) {
  if (observations.size() != 733 || !observations.allFinite()) throw std::runtime_error("Invalid SM observation");
  const auto start = std::chrono::steady_clock::now();
  vector_t action = OnnxPolicy::forward(observations);
  lastInferenceMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (!action.allFinite()) throw std::runtime_error("Non-finite SM action");
  action = action.cwiseMax(-actionLimit_).cwiseMin(actionLimit_);
  // LastActionObservationTerm must see post-clamp raw actions, never q_desired.
  outputTensors_.at(name2Index_.at("actions")).row(0) = action.cast<float>().transpose();
  lastObservation_ = observations;
  ++step_;
  return action;
}

SmCarryMotionCommand::SmCarryMotionCommand(std::shared_ptr<SmCarryReference> reference,
                                         std::shared_ptr<SmCarryOnnxPolicy> policy, bool alignHeading)
    : reference_(std::move(reference)), policy_(std::move(policy)), alignHeading_(alignHeading) {}
matrix3_t SmCarryMotionCommand::pelvisRotation() const {
  const auto index = model_->getPinModel().getFrameId("pelvis");
  if (index >= static_cast<size_t>(model_->getPinModel().nframes)) throw std::runtime_error("Pelvis frame missing");
  return model_->getPinData().oMf[index].rotation();
}
void SmCarryMotionCommand::reset() {
  alignment_.setIdentity();
  if (alignHeading_) {
    // Recover the first reference rotation from the 6D output at an identity pelvis.
    const auto first = reference_->goal(0, matrix3_t::Identity(), matrix3_t::Identity());
    matrix3_t rotation;
    rotation.col(0) << first[580], first[582], first[584];
    rotation.col(1) << first[581], first[583], first[585];
    rotation.col(2) = rotation.col(0).cross(rotation.col(1));
    alignment_ = yawRotation(pelvisRotation()) * yawRotation(rotation).transpose();
  }
}
vector_t SmCarryMotionCommand::getValue() {
  return reference_->goal(policy_->step(), pelvisRotation(), alignment_);
}

controller_interface::CallbackReturn SmCarryController::on_init() {
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS)
    return controller_interface::CallbackReturn::ERROR;
  auto_declare("sm_carry.control_path", std::string());
  auto_declare("sm_carry.reference_path", std::string());
  auto_declare("sm_carry.align_heading", true);
  auto_declare("sm_carry.debug", false);
  return controller_interface::CallbackReturn::SUCCESS;
}
controller_interface::CallbackReturn SmCarryController::on_configure(const rclcpp_lifecycle::State& state) {
  try {
    reference_ = std::make_shared<SmCarryReference>(get_node()->get_parameter("sm_carry.reference_path").as_string());
    smPolicy_ = std::make_shared<SmCarryOnnxPolicy>(get_node()->get_parameter("policy.path").as_string(),
                                                 get_node()->get_parameter("sm_carry.control_path").as_string());
    policy_ = smPolicy_; policy_->init();
    if (reference_->jointNames() != policy_->getJointNames()) throw std::runtime_error("Reference joint order mismatch");
    alignHeading_ = get_node()->get_parameter("sm_carry.align_heading").as_bool();
    debug_ = get_node()->get_parameter("sm_carry.debug").as_bool();
    if (debug_) debugPublisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("/sm_carry/debug", 10);
    return RlController::on_configure(state);
  } catch (const std::exception& error) {
    RCLCPP_ERROR(get_node()->get_logger(), "SM carry configuration failed: %s", error.what());
    return controller_interface::CallbackReturn::ERROR;
  }
}
controller_interface::CallbackReturn SmCarryController::on_activate(const rclcpp_lifecycle::State& state) {
  lastPublishedStep_ = std::numeric_limits<size_t>::max();
  return RlController::on_activate(state);
}
bool SmCarryController::parserCommand(const std::string& name) {
  if (name == "sm_carry_motion") {
    commandManager_->addTerm(std::make_shared<SmCarryMotionCommand>(reference_, smPolicy_, alignHeading_));
    return true;
  }
  return RlController::parserCommand(name);
}
controller_interface::return_type SmCarryController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  const auto result = RlController::update(time, period);
  if (debug_ && smPolicy_->step() && smPolicy_->step() != lastPublishedStep_) {
    lastPublishedStep_ = smPolicy_->step();
    std_msgs::msg::Float64MultiArray message;
    const auto model = leggedModel();
    const auto index = model->getPinModel().getFrameId("pelvis");
    const auto& pose = model->getPinData().oMf[index];
    const quaternion_t q(pose.rotation());
    message.data = {static_cast<double>(smPolicy_->step()-1), time.seconds(), smPolicy_->lastInferenceMs(),
                    static_cast<double>(reference_->size()), pose.translation().x(), pose.translation().y(),
                    pose.translation().z(), q.w(), q.x(), q.y(), q.z()};
    const auto& obs = smPolicy_->lastObservation();
    message.data.insert(message.data.end(), obs.data(), obs.data()+obs.size());
    const auto action = smPolicy_->getLastAction();
    message.data.insert(message.data.end(), action.data(), action.data()+action.size());
    debugPublisher_->publish(message);
  }
  return result;
}
}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::SmCarryController, controller_interface::ControllerInterface)
