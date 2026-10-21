## py-spy flamegraph: pyspy-4xl-full.svg

- Total samples: 5,859
- Frames: 337

### Top frames by self-time

| self | % of root | name | location |
|---:|---:|---|---|
| 884 | 15.1% | _step | api/physics_context/physics_context.py:565 |
| 381 | 6.5% | update | rsl_rl/algorithms/ppo.py:394 |
| 243 | 4.1% | synchronize | warp/context.py:6129 |
| 215 | 3.7% | update | rsl_rl/algorithms/ppo.py:292 |
| 208 | 3.6% | _engine_run_backward | torch/autograd/graph.py:824 |
| 203 | 3.5% | sample | torch/distributions/normal.py:74 |
| 122 | 2.1% | compute | isaaclab/managers/reward_manager.py:149 |
| 92 | 1.6% | reset | isaaclab/managers/reward_manager.py:118 |
| 88 | 1.5% | reset | isaaclab/managers/reward_manager.py:121 |
| 80 | 1.4% | compute | isaaclab/managers/reward_manager.py:156 |
| 79 | 1.3% | compute | isaaclab/actuators/actuator_pd.py:137 |
| 72 | 1.2% | joint_deviation_l1 | isaaclab/envs/mdp/rewards.py:178 |
| 70 | 1.2% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1812 |
| 67 | 1.1% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1825 |
| 62 | 1.1% | log | rsl_rl/runners/on_policy_runner.py:311 |
| 61 | 1.0% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1813 |
| 58 | 1.0% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1820 |
| 56 | 1.0% | _clip_effort | isaaclab/actuators/actuator_base.py:364 |
| 53 | 0.9% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1814 |
| 49 | 0.8% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1832 |
| 49 | 0.8% | forward | torch/nn/modules/linear.py:125 |
| 48 | 0.8% | sample | torch/distributions/normal.py:74 |
| 45 | 0.8% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1833 |
| 43 | 0.7% | _apply_actuator_model | isaaclab/assets/articulation/articulation.py:1821 |
| 37 | 0.6% | illegal_contact | isaaclab/envs/mdp/terminations.py:160 |

### Call tree (top branches, depth limited)

- **5859 (100.0%)** all
  - **5856 (99.9%)** <module> (benchmark_rsl_rl.py:258)
    - **5856 (99.9%)** wrapper (isaaclab_tasks/utils/hydra.py:104)
      - **5856 (99.9%)** decorated_main (hydra/main.py:94)
        - **5856 (99.9%)** _run_hydra (hydra/_internal/utils.py:394)
          - **5856 (99.9%)** _run_app (hydra/_internal/utils.py:457)
            - **5856 (99.9%)** run_and_report (hydra/_internal/utils.py:220)
              - **5856 (99.9%)** <lambda> (hydra/_internal/utils.py:458)
                - **5856 (99.9%)** run (hydra/_internal/hydra.py:119)
                  - **5856 (99.9%)** run_job (hydra/core/utils.py:186)
                    - **5856 (99.9%)** hydra_main (isaaclab_tasks/utils/hydra.py:101)
                      - **5856 (99.9%)** main (benchmark_rsl_rl.py:216)
                        - **4435 (75.7%)** learn (rsl_rl/runners/on_policy_runner.py:206)
                          - **4433 (75.7%)** step (isaaclab_rl/rsl_rl/vecenv_wrapper.py:176)
                            - **4433 (75.7%)** step (gymnasium/wrappers/common.py:393)
                              - **4433 (75.7%)** step (gymnasium/core.py:327)
                                - **923 (15.8%)** step (isaaclab/envs/manager_based_rl_env.py:190)
                                  - **922 (15.7%)** step (isaaclab/sim/simulation_context.py:635)
                                    - **922 (15.7%)** step (api/simulation_context/simulation_context.py:713)
                                      - **884 (15.1%)** _step (api/physics_context/physics_context.py:565)  *[leaf]*
                                      - *(1 more children below threshold)*
                                - **869 (14.8%)** step (isaaclab/envs/manager_based_rl_env.py:221)
                                  - **381 (6.5%)** _reset_idx (isaaclab/envs/manager_based_rl_env.py:364)
                                    - **326 (5.6%)** apply (isaaclab/managers/event_manager.py:244)
                                      - **61 (1.0%)** apply_external_force_torque (isaaclab/envs/mdp/events.py:835)
                                        - *(4 more children below threshold)*
                                      - *(14 more children below threshold)*
                                    - *(2 more children below threshold)*
                                  - **208 (3.6%)** _reset_idx (isaaclab/envs/manager_based_rl_env.py:377)
                                    - **92 (1.6%)** reset (isaaclab/managers/reward_manager.py:118)  *[leaf]*
                                    - **88 (1.5%)** reset (isaaclab/managers/reward_manager.py:121)  *[leaf]*
                                    - *(1 more children below threshold)*
                                  - **78 (1.3%)** _reset_idx (isaaclab/envs/manager_based_rl_env.py:358)
                                    - **75 (1.3%)** compute (isaaclab/managers/curriculum_manager.py:138)
                                      - *(3 more children below threshold)*
                                  - **76 (1.3%)** _reset_idx (isaaclab/envs/manager_based_rl_env.py:360)
                                    - **70 (1.2%)** reset (isaaclab/scene/interactive_scene.py:461)
                                      - *(4 more children below threshold)*
                                    - *(1 more children below threshold)*
                                  - **72 (1.2%)** _reset_idx (isaaclab/envs/manager_based_rl_env.py:383)
                                    - **71 (1.2%)** reset (isaaclab/managers/command_manager.py:353)
                                      - *(3 more children below threshold)*
                                  - *(2 more children below threshold)*
                                - **856 (14.6%)** step (isaaclab/envs/manager_based_rl_env.py:188)
                                  - **852 (14.5%)** write_data_to_sim (isaaclab/scene/interactive_scene.py:467)
                                    - **779 (13.3%)** write_data_to_sim (isaaclab/assets/articulation/articulation.py:214)
                                      - **186 (3.2%)** _apply_actuator_model (isaaclab/assets/articulation/articulation.py:1818)
                                        - **79 (1.3%)** compute (isaaclab/actuators/actuator_pd.py:137)  *[leaf]*
                                        - **61 (1.0%)** compute (isaaclab/actuators/actuator_pd.py:139)
                                          - *(1 more children below threshold)*
                                        - *(2 more children below threshold)*
                                      - **83 (1.4%)** _apply_actuator_model (isaaclab/assets/articulation/articulation.py:1820)
                                        - *(1 more children below threshold)*
                                      - **70 (1.2%)** _apply_actuator_model (isaaclab/assets/articulation/articulation.py:1812)  *[leaf]*
                                      - **67 (1.1%)** _apply_actuator_model (isaaclab/assets/articulation/articulation.py:1825)  *[leaf]*
                                      - **61 (1.0%)** _apply_actuator_model (isaaclab/assets/articulation/articulation.py:1813)  *[leaf]*
                                      - *(8 more children below threshold)*
                                    - *(3 more children below threshold)*
                                - **691 (11.8%)** step (isaaclab/envs/manager_based_rl_env.py:208)
                                  - **556 (9.5%)** compute (isaaclab/managers/reward_manager.py:149)
                                    - **72 (1.2%)** joint_deviation_l1 (isaaclab/envs/mdp/rewards.py:178)  *[leaf]*
                                    - **61 (1.0%)** feet_slide (isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py:79)
                                      - *(2 more children below threshold)*
                                    - *(15 more children below threshold)*
                                  - **80 (1.4%)** compute (isaaclab/managers/reward_manager.py:156)  *[leaf]*
                                  - *(4 more children below threshold)*
                                - **560 (9.6%)** step (isaaclab/envs/manager_based_rl_env.py:240)
                                  - **559 (9.5%)** compute (isaaclab/managers/observation_manager.py:268)
                                    - **518 (8.8%)** compute_group (isaaclab/managers/observation_manager.py:326)
                                      - **442 (7.5%)** height_scan (isaaclab/envs/mdp/observations.py:244)
                                        - **442 (7.5%)** data (isaaclab/sensors/ray_caster/ray_caster.py:100)
                                          - **425 (7.3%)** _update_outdated_buffers (isaaclab/sensors/sensor_base.py:353)
                                            - **308 (5.3%)** _update_buffers_impl (isaaclab/sensors/ray_caster/ray_caster.py:296)
                                              - **248 (4.2%)** raycast_mesh (isaaclab/utils/warp/ops.py:118)
                                                - **243 (4.1%)** synchronize (warp/context.py:6129)  *[leaf]*
                                              - *(2 more children below threshold)*
                                            - *(4 more children below threshold)*
                                          - *(1 more children below threshold)*
                                      - *(4 more children below threshold)*
                                    - *(1 more children below threshold)*
                                - *(7 more children below threshold)*
                        - **1039 (17.7%)** learn (rsl_rl/runners/on_policy_runner.py:262)
                          - **381 (6.5%)** update (rsl_rl/algorithms/ppo.py:394)  *[leaf]*
                          - **221 (3.8%)** update (rsl_rl/algorithms/ppo.py:260)
                            - **203 (3.5%)** act (rsl_rl/modules/actor_critic.py:122)
                              - **203 (3.5%)** sample (torch/distributions/normal.py:74)  *[leaf]*
                            - *(1 more children below threshold)*
                          - **215 (3.7%)** update (rsl_rl/algorithms/ppo.py:292)  *[leaf]*
                          - **208 (3.6%)** update (rsl_rl/algorithms/ppo.py:375)
                            - **208 (3.6%)** backward (torch/_tensor.py:648)
                              - **208 (3.6%)** backward (torch/autograd/__init__.py:353)
                                - **208 (3.6%)** _engine_run_backward (torch/autograd/graph.py:824)  *[leaf]*
                          - *(1 more children below threshold)*
                        - **234 (4.0%)** learn (rsl_rl/runners/on_policy_runner.py:204)
                          - **208 (3.6%)** act (rsl_rl/algorithms/ppo.py:142)
                            - **150 (2.6%)** act (rsl_rl/modules/actor_critic.py:121)
                              - **76 (1.3%)** update_distribution (rsl_rl/modules/actor_critic.py:109)
                                - **74 (1.3%)** _wrapped_call_impl (torch/nn/modules/module.py:1751)
                                  - **74 (1.3%)** _call_impl (torch/nn/modules/module.py:1762)
                                    - **73 (1.2%)** forward (torch/nn/modules/container.py:240)
                                      - **67 (1.1%)** _wrapped_call_impl (torch/nn/modules/module.py:1751)
                                        - **59 (1.0%)** _call_impl (torch/nn/modules/module.py:1762)
                                          - *(2 more children below threshold)*
                              - *(2 more children below threshold)*
                            - *(1 more children below threshold)*
                          - *(1 more children below threshold)*
                        - **113 (1.9%)** learn (rsl_rl/runners/on_policy_runner.py:270)
                          - **62 (1.1%)** log (rsl_rl/runners/on_policy_runner.py:311)  *[leaf]*
                          - *(3 more children below threshold)*
                        - *(1 more children below threshold)*

