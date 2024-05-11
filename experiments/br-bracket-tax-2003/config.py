from ai_economist import foundation

ENV_CONFIG = {"components": ""}


""" 
components:
  - Build:
      build_labor: 10
      payment: 10
      payment_max_skill_multiplier: 3
      skill_dist: pareto
  - ContinuousDoubleAuction:
      max_bid_ask: 10
      max_num_orders: 5
      order_duration: 50
      order_labor: 0.25
  - Gather:
      collect_labor: 1
      move_labor: 1
      skill_dist: pareto
  - PeriodicBracketTax:
      bracket_spacing: us-federal
      disable_taxes: true
      period: 100
      tax_annealing_schedule:
      - -100
      - 0.001
      usd_scaling: 1000
  dense_log_frequency: 20
  energy_cost: 0.21
  energy_warmup_constant: 10000
  energy_warmup_method: auto
  env_layout_file: quadrant_25x25_20each_30clump.txt
  episode_length: 1000
  fixed_four_skill_and_loc: true
  flatten_masks: true
  flatten_observations: true
  isoelastic_eta: 0.23
  multi_action_mode_agents: false
  multi_action_mode_planner: true
  n_agents: 4
  planner_gets_spatial_info: false
  planner_reward_type: coin_eq_times_productivity
  scenario_name: layout_from_file/simple_wood_and_stone
  starting_agent_coin: 0
  world_size:
  - 25
  - 25
  """
