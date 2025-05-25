# Water Management Simulation

This repository implements an agent-based evolutionary game to study water resource allocation under varying environmental and policy scenarios. It models multiple actors who decide whether to cooperate or defect in water usage, tracks ecological and economic impacts, and provides tools for running simulations and visualizing results.

## Usage
### Single Scenario (Notebook)
Open single_scenario.ipynb to run and customize one simulation interactively.

### Multi-Scenario Analysis (Notebook)
Open multi_scenarios.ipynb for comparative visualizations and advanced analysis.

## Installation

```shell

# Clone the repository:
   git clone https://github.com/Iyeleon/Hackathon-Water-Scarcity.git

# Install dependencies:
   pip install -r requirements.txt


# Project Structure
    ├── parameters/
    │   ├── data.csv               # Real riverflow time series
    │   └── scenarios/             # YAML parameter files for scenarios
    ├── src/
    │   ├── policies/
    │   │   ├── custom_policies.py # policy function templates
    │   ├── core.py                # Main WaterManagementSimulation class
    │   ├── actors.py              # ActorManager: decision-making & learning
    │   ├── water_allocation.py    # WaterAllocator: pumping & quota logic
    │   ├── ecology.py             # EcologyManager: flow & impact calculations
    │   ├── utils.py               # Helper functions (e.g., YAML loader)
    │   ├── plot_analysis.py       # Time-series plots for individual runs
    │   ├── scenarios.py           # Script to batch-run scenarios
    │   ├── fast_scenarios.py      # Parallelized Script to batch-run scenarios
    │   ├── policy_regulation.py   # Main PolicyRegulatorClass: optimize policies
    │   └── plot_multi_analysis.py # Impact trade-off & correlation plots 
    ├── policy_optimization.ipynb      # Interactive demo for one scenario
    ├── single_scenario.ipynb      # Interactive demo for one scenario
    ├── multi_scenarios.ipynb      # Comparative analysis notebook
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Project overview and usage guide

# How to run
- run policy_optimization.ipynb to optimize policy (takes ~8 hours to run). Change optimize to `True` to run. if False, skips training. Loads preoptimized `PolicyRegulator`
run multi_scenarios_trial_33_best.ipynb to generate quota and incentives function and run simulation across multiple scenarios
run single_scenario_* to see effect of policies in single scenario environment