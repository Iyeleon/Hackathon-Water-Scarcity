import os
import src.utils as utils
import src.core as wms
import pandas as pd
import numpy as np
from types import MethodType
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_simulation(params, custom_quota, custom_incentive_policy, scenario, station, verbose = 2):
    # Initialize simulation
    simulation = wms.WaterManagementSimulation(**params, verbose = verbose)
    
    # Set custom policies
    simulation.incentive_policy = MethodType(custom_incentive_policy, simulation)
    simulation.compute_actor_quota = MethodType(custom_quota, simulation)
    
    # Run simulation
    simulation.run_simulation()
    
    # Get scores
    ecological_impact, economic_impact, priority_ok = simulation.get_final_scores_scaled()

    # Calculate raw ecological impact (total number of breaches)
    raw_ecol_impact = np.sum(simulation.w_ecol_impact > 0)
    raw_econ_impact = np.sum(simulation.h_econ_impacts)
    
    # Calculate average percentage of cooperators
    # Average over all iterations and turns
    avg_coop = np.mean(simulation.h_actions)

    if verbose >= 2:
        log = f"Scenario: {scenario}, Station: {station}, Scarcity: {params['scarcity']}, Bias: {params['global_forecast_bias']}, \
Uncertainty:{params['global_forecast_uncertainty']}, Eco Impact: {ecological_impact:.3f}, Econ Impact: {economic_impact:.3f}, \
Raw Eco Impact: {raw_ecol_impact:.1f}, Cooperation %: {avg_coop*100:.1f}%"
    else:
        log = None
    
    return log, {
        'ecological_impact': ecological_impact, 
        'economic_impact': economic_impact, 
        'raw_ecological_impact': raw_ecol_impact, 
        'raw_economic_impact': raw_econ_impact,
        'cooperation_percentage': avg_coop,
        'priority_ok': priority_ok
    }
    
def fast_run_all_scenarios(
        turns: int,
        iterations: int,
        custom_incentive_policy: Callable,
        custom_quota: Callable,
        scenarios: list = None,
        scarcity_levels: list = None,
        exploration_biases: list = None,
        exploration_uncertainties: list = None,
        max_workers:int = 8,
        verbose = 2
):
    """
    Run simulations across all defined scenarios and return a DataFrame with results.
    """
    # Scenario parameters
    if scenarios is None:
        scenarios = ["0.yml", "1.yml", "0-v.yml", "1-v.yml", 
                    "0-b.yml", "1-b.yml", "0-c.yml", "1-c.yml"]
    if scarcity_levels is None:
        scarcity_levels = ["low", "medium", "high"]
    if exploration_biases is None:
        exploration_biases = [0.0, 0.25, -0.25, 0.5, -0.5]  
    if exploration_uncertainties is None:
        exploration_uncertainties = [0.0, 0.25, 0.5]
        
    # Station definitions
    stations = {
        1: {'station': 6125320, 'DOE': 0.1, 'DCR': 0.05},  # Small river basin
        2: {'station': 6124501, 'DOE': 7, 'DCR': 3.5}      # Large river basin
    }

    if verbose >= 1:
        print("Starting simulations across all scenarios...")
        print('Generating scenarios configurations...')
    
    # get all param sets
    params_set = []
    for station in stations:
        for scarcity in scarcity_levels:
            for scenario in scenarios:
                # Determine whether to vary bias and uncertainty based on scenario name
                if len(scenario) == 5: # Base scenarios
                    biases = exploration_biases
                    uncertainties = exploration_uncertainties
                else:  # Variant scenarios
                    biases = [0.0]
                    uncertainties = [0.0]
                
                for bias in biases:
                    for uncertainty in uncertainties:
                        # Configure simulation
                        yaml_path = f'parameters/scenarios/{scenario}'  
                        params = utils.load_parameters_from_yaml(yaml_path)
                        params["total_turns"] = turns
                        params["nb_iterations"] = iterations
                        params["scarcity"] = scarcity
                        params["global_forecast_bias"] = bias
                        params["global_forecast_uncertainty"] = uncertainty
                        params["station"] = stations[station]["station"]
                        params["DOE"] = stations[station]["DOE"]
                        params["DCR"] = stations[station]["DCR"]
                        params_set.append([params, custom_quota, custom_incentive_policy, scenario, station, verbose])

    if verbose >= 1:
        print('Scenarios configurations complete!')
        print(f'There are {len(params_set)} unique configurations')
        print('Starting simulations')
    # # run in parallel
    # with ProcessPoolExecutor(max_workers = max_workers) as executor:
    #     futures = [executor.submit(run_single_simulation, *args) for args in params_set]
    #     outputs = [f.result() for f in futures]

    outputs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_simulation, *args) for args in params_set]
    
        for future in as_completed(futures):
            log, result = future.result()
            if verbose >= 2:
                print(log)  # This prints in the main process
            outputs.append(result)

    # collect all result in a dataframe
    results = []
    for idx, param in enumerate(params_set):
        data = {k:v for k, v in param[0].items() if k in ['global_forecast_bias', 'global_forecast_uncertainty', 'scarcity', 'station']}
        data['scenario'] = param[3]
        data['station'] = param[4]
        for k, v in outputs[idx].items():
            data[k] = v
        results.append(data)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns = {'global_forecast_bias': 'bias', 'global_forecast_uncertainty': 'uncertainty'})
    
    # Add color mappings
    scarcity_colors = {"low": "yellow", "medium": "orange", "high": "red"}
    scenario_colors = {
        "0.yml": "blue", "1.yml": "red", 
        "0-v.yml": "purple", "1-v.yml": "orange",
        "0-b.yml": "blue", "1-b.yml": "red", 
        "0-c.yml": "blue", "1-c.yml": "red"
    }
    scenario_names = {
        "0.yml": "0", "1.yml": "1", 
        "0-v.yml": "0-v", "1-v.yml": "1-v",
        "0-b.yml": "0-b", "1-b.yml": "1-b", 
        "0-c.yml": "0-c", "1-c.yml": "1-c"
    }
    station_colors = {"1": "green", "2": "blue"}
    
    results_df["scarcity_color"] = results_df["scarcity"].map(scarcity_colors)
    results_df["scenario_color"] = results_df["scenario"].map(scenario_colors)
    results_df["scenario_name"] = results_df["scenario"].map(scenario_names)
    results_df["station_color"] = results_df["station"].astype(str).map(station_colors)
    
    if verbose >= 1:
        print(f"Completed {len(results_df)} simulation runs")
    
    return results_df