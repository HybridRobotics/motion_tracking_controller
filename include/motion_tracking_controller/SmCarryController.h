#pragma once

#include <legged_rl_controllers/OnnxPolicy.h>
#include <legged_rl_controllers/RlController.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <limits>

namespace legged {

// Deployment-side reference lookup. No box state or world model is used.
class SmCarryReference {
 public:
  explicit SmCarryReference(const std::string& path);
  vector_t goal(size_t step, const matrix3_t& pelvisRotation,
                const matrix3_t& referenceAlignment) const;
  size_t size() const { return positions_.rows(); }
  const std::vector<std::string>& jointNames() const { return jointNames_; }

 private:
  matrix_t positions_, velocities_;
  std::vector<matrix3_t> rotations_;
  std::vector<std::string> jointNames_;
  std::vector<size_t> offsets_;
};

class SmCarryOnnxPolicy : public OnnxPolicy {
 public:
  SmCarryOnnxPolicy(const std::string& path, const std::string& controlPath);
  void reset() override;
  vector_t forward(const vector_t& observations) override;
  size_t step() const { return step_; }
  const vector_t& lastObservation() const { return lastObservation_; }
  double lastInferenceMs() const { return lastInferenceMs_; }

 protected:
  void parseMetadata() override;
  void checkInputOutput() override;

 private:
  nlohmann::json control_;
  size_t step_{0};
  double actionLimit_{20.0}, lastInferenceMs_{0};
  vector_t lastObservation_;
};

class SmCarryMotionCommand : public CommandTerm {
 public:
  SmCarryMotionCommand(std::shared_ptr<SmCarryReference> reference,
                       std::shared_ptr<SmCarryOnnxPolicy> policy, bool alignHeading);
  vector_t getValue() override;
  size_t getSize() const override { return 640; }
  void reset() override;

 private:
  matrix3_t pelvisRotation() const;
  std::shared_ptr<SmCarryReference> reference_;
  std::shared_ptr<SmCarryOnnxPolicy> policy_;
  bool alignHeading_;
  matrix3_t alignment_{matrix3_t::Identity()};
};

class SmCarryController : public RlController {
 public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;
  controller_interface::return_type update(const rclcpp::Time&, const rclcpp::Duration&) override;

 protected:
  bool parserCommand(const std::string& name) override;

 private:
  std::shared_ptr<SmCarryReference> reference_;
  std::shared_ptr<SmCarryOnnxPolicy> smPolicy_;
  bool alignHeading_{true}, debug_{false};
  size_t lastPublishedStep_{std::numeric_limits<size_t>::max()};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debugPublisher_;
};

}  // namespace legged
