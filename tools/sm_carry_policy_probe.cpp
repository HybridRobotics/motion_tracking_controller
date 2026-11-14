#include "motion_tracking_controller/SmCarryController.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: sm_carry_policy_probe ASSET_DIRECTORY");
    const std::string root = argv[1];
    std::ifstream file(root+"/fixtures.json"); const auto fixtures=nlohmann::json::parse(file);
    legged::SmCarryReference reference(root+"/reference.json");
    double maxGoalError=0;
    for (const auto& test : fixtures.at("goals")) {
      legged::matrix3_t rotation;
      for (int i=0;i<9;++i) rotation(i/3,i%3)=test.at("rotation").at(i).get<double>();
      const auto value=reference.goal(test.at("step").get<size_t>(),rotation,legged::matrix3_t::Identity());
      for(int i=0;i<640;++i)maxGoalError=std::max(maxGoalError,std::abs(value[i]-test.at("goal").at(i).get<double>()));
    }
    if(maxGoalError>1e-9)throw std::runtime_error("Future goal mismatch vs NumPy reference");
    legged::SmCarryOnnxPolicy policy(root+"/actor.onnx",root+"/control.json");
    policy.init();policy.reset();legged::vector_t obs(733);
    for(int i=0;i<733;++i)obs[i]=fixtures.at("obs").at(i).get<double>();
    auto action=policy.forward(obs);double maxActionError=0;
    for(int i=0;i<29;++i)maxActionError=std::max(maxActionError,std::abs(action[i]-std::clamp(fixtures.at("actions").at(i).get<double>(),-20.,20.)));
    if(maxActionError>1e-4)throw std::runtime_error("C++ ONNX action mismatch vs supplied smoke fixture");
    if((policy.getLastAction()-action).norm()>1e-6)throw std::runtime_error("Previous action is not post-clamp raw action");
    std::vector<double> times;
    for(int i=0;i<100;++i){policy.forward(obs);times.push_back(policy.lastInferenceMs());}
    std::sort(times.begin(),times.end());
    policy.reset();if(policy.step()!=0||policy.getLastAction().norm()!=0)throw std::runtime_error("Reset failed");
    std::cout<<"SM_PROBE_PASS goal_max_error="<<maxGoalError<<" action_max_error="<<maxActionError
             <<" median_ms="<<times[50]<<" p95_ms="<<times[95]<<" max_ms="<<times.back()<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"SM_PROBE_FAIL "<<e.what()<<std::endl;return 1;}
}
