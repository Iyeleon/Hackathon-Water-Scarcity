import os
import json
import pickle
from functools import wraps, partial
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.samplers import NSGAIISampler, NSGAIIISampler, TPESampler, RandomSampler, QMCSampler
import pymoo
from abc import ABC, abstractmethod
import src.fast_scenarios as fsc
import matplotlib.pyplot as plt
from plotly.io import show

plt.style.use('ggplot') 

# Enable grid by default
plt.rcParams['axes.grid'] = True

# Set default figure size to 15x5 inches
plt.rcParams['figure.figsize'] = (15, 5)

# Optional: make grid more visible
plt.rcParams['grid.color'] = '#ffffff'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.9


def apply_parameter(**kwargs):
    def wrapper(func):
        def inner_func():
            return partial(func, **kwargs)
        return inner_func
    return wrapper
            
class KwargWrapper:
    def __init__(self, func, **preset_kwargs):
        self.func = func
        self.preset_kwargs = preset_kwargs

    def __call__(self, *args, **kwargs):
        combined_kwargs = {**self.preset_kwargs, **kwargs}
        return self.func(*args, **combined_kwargs)
    
class BasePolicyRegulator(ABC):
    def __init__(
        self,
        quota_policy,
        incentives_policy,
        quota_params,
        incentives_params,
        simulation_params,
        name = 'multi-objective',
    ):
        self.quota_policy = quota_policy
        self.incentives_policy = incentives_policy
        self.quota_params = quota_params
        self.incentives_params = incentives_params
        self.simulation_params = simulation_params
        self.name = name or f'{sampler.__class__.__name__}'
        self.directions = ['minimize', 'maximize']#, 'maximize']
        self.obj_names = ['Ecological Impact', 'Economic Impact', 'Satisfaction']
        self.optimized = False

        @abstractmethod
        def optimize_objective(self):
            raise NotImplementedError

class PolicyRegulator(BasePolicyRegulator):
    samplers = {
        'random' : RandomSampler(seed = 42),
        'tpe': TPESampler(seed = 42),
        'nsga_2': NSGAIISampler(population_size = 10, seed = 42),
        'nsga_3': NSGAIIISampler(population_size = 10, seed = 42),
        'qmc': QMCSampler(seed = 42)
    }
    
    def __init__(self, *args, sampler = 'nsga_2',**kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(sampler, str):
            # if str sampler provided
            self.sampler = self.samplers[sampler]
        else:
            # else, use default from the available samplers
            self.sampler = sampler or self.samplers['ngsa_2']
        self.study = None
        self.cooperation_rate = []
        
    def parse_policy_params(self, trial, policy_params):
        hyp = {}
        for param, param_range in policy_params.items():
            if isinstance(param_range, (list, tuple)):
                dtype = param_range[0]
                inf = param_range[1]
                sup = param_range[2]
                if dtype ==  'int':
                    hyp[param] = trial.suggest_int(param, inf, sup)
                elif dtype == 'float':
                    hyp[param] = trial.suggest_float(param, inf, sup)
                else:
                    raise ValueError(f"dtype must be either a float or int. got type {dtype}")
            else:
                hyp[param] = param_range
                                    
        return hyp

    def get_objective(self):
        def objective(trial):
            # define hyperparameter space
            quota_hyp = self.parse_policy_params(trial, self.quota_params)
            incentives_hyp = self.parse_policy_params(trial, self.incentives_params)
    
            # define policy functions
            quota_policy = KwargWrapper(self.quota_policy, **quota_hyp)
            incentives_policy = KwargWrapper(self.incentives_policy, **incentives_hyp)
    
            # run simulations
            results = fsc.fast_run_all_scenarios(custom_quota = quota_policy, custom_incentive_policy = incentives_policy, **self.simulation_params)
    
            # get metrics
            ecological_impact = results.ecological_impact.mean()
            economic_impact = results.economic_impact.mean()
            raw_ecological_impact = results.raw_ecological_impact.mean()
            cooperation_percentage = results.cooperation_percentage.mean()
            priority_ok = results.priority_ok.mean()
            
            trial.set_user_attr('cooperation', cooperation_percentage)
            trial.set_user_attr('satisfaction', priority_ok)

            penalty = (1 - priority_ok) * 5
            ecological_impact += penalty
            economic_impact -= penalty

    
            # return results
            return ecological_impact, economic_impact#, priority_ok
        return objective
        
    def optimize_objective(self, n_trials = 50, timeout = 900):
        # create study
        study = optuna.create_study(directions = self.directions, sampler = self.sampler, study_name = self.name)

        print('Starting optimization...')
        #optimize
        study.optimize(self.get_objective(), n_trials = n_trials, timeout = timeout)
        
        self.study = study
        self.optimized = True

        print('Optimization complete')

    def _check_is_optimized(self):
        assert self.optimized, 'Objective is not optimized. Call model.optimize_objective()'

    def _get_is_dominated(self, candidate, population):
        for other in population:
            if np.allclose(candidate, other):
                continue  # Don't compare candidate to itself
    
            better_or_equal = True
            strictly_better = False
    
            for c_val, o_val, direction in zip(candidate, other, self.directions):
                if direction == 'minimize':
                    if o_val > c_val:
                        better_or_equal = False
                        break
                    elif o_val < c_val:
                        strictly_better = True
                elif direction == 'maximize':
                    if o_val < c_val:
                        better_or_equal = False
                        break
                    elif o_val > c_val:
                        strictly_better = True
                else:
                    raise ValueError(f"Invalid direction: {direction} (must be 'minimize' or 'maximize')")
    
            if better_or_equal and strictly_better:
                return True  # candidate is dominated
    
        return False  # candidate is not dominated

    def _get_satisfactory_best_trials(self, satisfaction_threshold = 0.99):
        best_trials = self.study.trials
        satisfactory_best_trials = [i for i in best_trials if i.values[2] >= satisfaction_threshold and i.values[1] > 0]
        return satisfactory_best_trials
        
        
    def plot_pareto_front(self, show_dominated=False, satisfaction_threshold = 0.99, **kwargs):
        # Get Pareto front trials
        
        pareto_front = self.study.best_trials
        # pareto_front = [i for i in self.study.best_trials if i.values[1] > 0]
        pareto_points = [trial.values for trial in pareto_front]


        # Sort Pareto points by x-value (objective 0)
        pareto_points.sort(key=lambda x: x[0])
    
        # Collect dominated trials if requested
        dominated_points = []
        if show_dominated:
            pareto_ids = {t._trial_id for t in pareto_front}
            dominated_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t._trial_id not in pareto_ids]
            dominated_points = [t.values for t in dominated_trials if t.values is not None]
    
        # Start plotting
        fig, ax = plt.subplots(figsize=(15, 7))
    
        # Plot dominated trials (if requested)
        if show_dominated and dominated_points:
            dom = list(zip(*dominated_points))
            plt.scatter(dom[0], dom[1], marker = 'o', facecolors = 'none', edgecolors = 'black', label='Dominated', alpha = 0.9)
    
        # Plot Pareto front
        pf = list(zip(*pareto_points))
        im = plt.scatter(pf[0], pf[1], s = 50, cmap = 'Reds', edgecolors = '#ffffff', linewidth = 0.75, vmin = 0.0, vmax = 1, label='Pareto Front')#, c = pf[2],)
        plt.plot(pf[0], pf[1], linestyle = '--', linewidth = 1.0)
        # cbar = fig.colorbar(im)
        # cbar.ax.set_ylabel('Satisfaction level')
        
    
        plt.xlabel(self.obj_names[0])
        plt.ylabel(self.obj_names[1])
        plt.title(f'{self.name}_Pareto Front')

    def find_best_pareto_elbow_angle(self, satisfaction_threshold = 0.99, plot=False):
        # # Get Pareto trials meeting satisfaction threshold
        # all_pareto_trials = self._get_satisfactory_best_trials(satisfaction_threshold)
        # all_values = np.array([t.values[:2] for t in all_pareto_trials])
    
        # # Filter out dominated points
        # is_dominated = [self._get_is_dominated(v, all_values) for v in all_values]
        # pareto_trials = [all_pareto_trials[i] for i in range(len(is_dominated)) if not is_dominated[i]]
        # values = all_values[~np.array(is_dominated)]
        
        pareto_trials = self.study.best_trials
        values = np.array([t.values[:2] for t in pareto_trials])
    
        # Manual fallback if too few points
        if len(pareto_trials) == 1:
            print("Only one Pareto-optimal trials found. Returning single solution.")
            return pareto_trials[0]
            
        if len(pareto_trials) <= 2:
            print("Only two Pareto-optimal trials found. Manual selection required.")
            return pareto_trials  # Let caller decide
                
        # sort values by first objective
        if self.directions[0] == 'minimize':
            sorting = np.argsort(values[:, 0])
        else:
            sorting = np.argsort(-values[:, 0])
        values = values[sorting]
        pareto_trials = np.array(pareto_trials)[sorting]
        dominated = values
    
        # Normalize for scale invariance
        values_norm = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-8)

        # compute finite differences
        diffs = np.diff(values_norm, prepend = 0, append = 0, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
        unit_vecs = diffs / norms


        # Compute angles between successive gradient vectors
        dot_products = np.sum(unit_vecs[:-1] * unit_vecs[1:], axis=1)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Angle in radians


        # Elbow is point with largest direction change
        elbow_idx = np.argmax(angles) + 1  # +1 to align with the middle point of 3

        if len(angles) > 3:
            for i in range(1, len(angles)):
                if angles[i] < angles[i - 1]:  # Angle stopped increasing
                    elbow_idx = i
                    break
            elbow_idx += 1  # align with middle point of three
    
        if plot:
            self._plot_pareto_optimization(values, angles, elbow_idx, method = 'Angle')

        return pareto_trials[elbow_idx]
        
    def find_best_pareto_elbow_gradient(self, satisfaction_threshold=0.99, plot=False):    
        # # Get Pareto trials meeting satisfaction threshold
        # all_pareto_trials = self._get_satisfactory_best_trials(satisfaction_threshold)
        # all_values = np.array([t.values[:2] for t in all_pareto_trials])
    
        # # Filter out dominated points
        # is_dominated = [self._get_is_dominated(v, all_values) for v in all_values]
        # pareto_trials = [all_pareto_trials[i] for i in range(len(is_dominated)) if not is_dominated[i]]
        # values = all_values[~np.array(is_dominated)]

        pareto_trials = self.study.best_trials
        values = np.array([t.values[:2] for t in pareto_trials])
    
        # Manual fallback if too few points
        if len(pareto_trials) == 1:
            print("Only one Pareto-optimal trials found. Returning single solution.")
            return pareto_trials[0]
            
        if len(pareto_trials) <= 2:
            print("Only two Pareto-optimal trials found. Manual selection required.")
            return pareto_trials  # Let caller decide
    
        # Sort points by first objective
        if self.directions[0] == 'minimize':
            sorting = np.argsort(values[:, 0])
        else:
            sorting = np.argsort(-values[:, 0])
        values = values[sorting]
        pareto_trials = np.array(pareto_trials)[sorting]
    
        # Normalize objective values
        values_norm = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-8)
    
        # Compute gradients (L2 norms of differences)
        diffs = np.diff(values_norm, prepend = 0, append = 0, axis=0)
        gradients = diffs[:, 1] / (diffs[:, 0] + 1e-8)
        elbow_idx = np.argmin(gradients)
        # note that elbow idx is already pareto idx - 1
    
        if plot:
            self._plot_pareto_optimization(values, gradients, elbow_idx, method="gradient")
    
        return pareto_trials[elbow_idx]
    
    def _plot_pareto_optimization(self, values_norm, angles, elbow_idx, method = 'Gradient'):
        print(angles)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Pareto front
        ax[0].plot(values_norm[:, 0], values_norm[:, 1], 'o-', label='Pareto Front')
        ax[0].axvline(values_norm[elbow_idx, 0], color='red', linestyle='--', label="Elbow")
        ax[0].set_xlabel(self.obj_names[0])
        ax[0].set_ylabel(self.obj_names[1])
        ax[0].set_title("Pareto Front with Elbow")
        ax[0].legend()

        # Plot changes
        ax[1].plot(range(1, len(angles)+1), angles, marker='o')
        ax[1].axvline(elbow_idx + 1, color='red', linestyle='--', label="Elbow")
        ax[1].set_xlabel("Index")
        ax[1].set_ylabel(f"{method.title()}")
        ax[1].set_title(f"Economic / Ecological Impact {method.title()}")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_hypervolume(self, figsize = None, save_path = None):
        ax = optuna.visualization.matplotlib.plot_hypervolume_history(self.study, [1.0, 0.0])#, 0.0])
        if figsize:
            fig = ax.get_figure()
            fig.set_size_inches(figsize)

        # change color
        for child in ax.get_children():
            if isinstance(child, plt.matplotlib.lines.Line2D):
                child.set_color('red') 
        if save_path is not None:
            plt.savefig(save_path)

    def plot_feature_importances(self, importance_type="fanova", target = None, save_path = None):
        # compute parameter importances
        importances = optuna.importance.get_param_importances(self.study, evaluator=importance_type, target = target)
    
        # plotting
        if not importances:
            print("No importances could be calculated.")
            return
    
        params = list(importances.keys())
        values = list(importances.values())
    
        plt.figure(figsize=(10, 6))
        plt.barh(params, values)#, color="red")
        plt.xlabel("Importance")
        plt.title(f"Hyperparameter Importances ({importance_type})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def save(self, save_path):
        os.makedirs(save_path)
        # save trial
        with open(os.path.join(save_path, 'study.pkl'), 'wb') as file:
            pickle.dump(self.study, file)

        # TO-DO fix function pickling error that happens occassionally
        # TEMP FIX - save policy reg without functions, 
        # # pickle quota function    
        # with open(os.path.join(save_path, 'quota_fn.pkl'), 'wb') as file:
        #      pickle.dump(self.quota_policy, file)
            
        # # pickle incentives function
        # with open(os.path.join(save_path, 'incentives_fn.pkl'), 'wb') as file:
        #      pickle.dump(self.incentives_policy, file)
        
        # save metadata       
        metadata = {
            'quota_params': self.quota_params,
            'incentives_params': self.incentives_params,
            'simulation_params': self.simulation_params,
            'name': self.name,
        }
        with open(os.path.join(save_path, 'metadata.json'), 'w') as file:
            json.dump(metadata, file)

    @classmethod
    def from_preoptimized(cls, load_path, quota_policy = None, incentives_policy = None):
        # TO-DO fix function pickling error that happens occassionally
        # TEMP FIX - load policy reg without functions, supply policy functions after loading,
         
        # with open(os.path.join(load_path, 'quota_fn.pkl'), 'rb') as file:
        #     quota_policy = pickle.load(file)

        # with open(os.path.join(load_path, 'incentives_fn.pkl'), 'rb') as file:
        #     incentives_policy = pickle.load(file)

        with open(os.path.join(load_path, 'study.pkl'), 'rb') as file:
            study = pickle.load(file)

        with open(os.path.join(load_path, 'metadata.json'), 'r') as file:
            metadata = json.load(file)

        model = cls(quota_policy, incentives_policy, **metadata)
        model.study = study

        model.optimized = True
        return model
    
    def get_best_policy_functions(self, params):
        # parse params
        quota_params = {}
        incentives_params = {}
        for k, v in params.items():
            if k in self.quota_params.keys():
                quota_params[k] = v
            elif k in self.incentives_params.keys():
                incentives_params[k] = v

        quota_function = KwargWrapper(self.quota_policy, **quota_params)
        incentives_function = KwargWrapper(self.incentives_policy, **incentives_params)

        return quota_function, incentives_function
        
                
        
        
            