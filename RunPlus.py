import random
import pickle
import pandas as pd
import hydra
from hydra.utils import instantiate, call
from omegaconf import OmegaConf, DictConfig
import mlflow
from RunManager import algo_data_multi
import Algorithms, FitnessFunctions, ExperimentsHelpers
import random
import numpy as np
import pandas as pd
import concurrent.futures
import itertools
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Type

from deap import tools

from Algorithms import *

# -------------------------------
# Run Functions
# -------------------------------

def hydra_algo_data_single(prob_info: Dict[str, Any], 
                          algo_config: DictConfig, 
                          algo_params: Dict[str, Any], 
                          seed: int) -> Dict[str, Any]:
    """
    """
    # Set the random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create and run the algorithm instance.
    algo_instance = instantiate(algo_config, **algo_params)
    algo_instance.run()  # This updates the instance's internal data.
    
    # Retrieve derived data from the run.
    unique_sols, unique_fits, noisy_fits, sol_iterations, sol_transitions = algo_instance.get_trajectory_data()
    seed_signature = algo_instance.seed_signature
    
    return {
        "problem_name": prob_info['name'],
        "problem_type": prob_info['type'],
        "problem_goal": prob_info['goal'],
        "dimensions": prob_info['dimensions'],
        "opt_global": prob_info['opt_global'],
        "mean_value": prob_info['mean_value'],
        "mean_weight": prob_info['mean_weight'],
        'PID': prob_info['PID'],
        "fit_func": algo_params['fitness_function'][0].__name__,
        "noise": algo_params['fitness_function'][1]['noise_intensity'],
        # "algo_class": algorithm_class.__name__,
        "algo_type": algo_instance.type,
        "algo_name": algo_instance.name,
        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        "stop_trigger": algo_instance.stop_trigger,
        "n_unique_sols": len(unique_sols),
        "unique_sols": unique_sols,
        "unique_fits": unique_fits,
        "noisy_fits": noisy_fits,
        "final_fit": unique_fits[-1],
        "max_fit": max(unique_fits),
        "min_fit": min(unique_fits),
        "sol_iterations": sol_iterations,
        "sol_transitions": sol_transitions,
        "seed": seed,
        "seed_signature": seed_signature,
    }

def hydra_algo_data_multi(prob_info: Dict[str, Any],
                    algo_config: DictConfig, 
                    algo_params: Dict[str, Any], 
                    num_runs: int, 
                    base_seed: int = 0, 
                    parallel: bool = False) -> pd.DataFrame:

    results_list = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_runs):
                seed = base_seed + i
                futures.append(executor.submit(hydra_algo_data_single, prob_info, algo_config, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results_list.append(hydra_algo_data_single(prob_info, algo_config, algo_params, seed))
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by='seed')
    return df_sorted

# -------------------------------
# Hydra config management
# -------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialise MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Load problem if required
    if "loader" in cfg.problem:
        outputs = call(cfg.problem.loader)
        n_items, capacity, optimal, values, weights, items_dict, _ = outputs
        
        items_dict = {
        int(k): (float(v[0]), float(v[1]))
        for k, v in items_dict.items()
        }

        cfg.problem.dimensions = int(n_items)
        cfg.problem.opt_global = float(optimal)
        cfg.problem.capacity = float(capacity)
        cfg.problem.mean_value = float(np.mean(values))
        cfg.problem.mean_weight = float(np.mean(weights))
        cfg.problem.items_dict = items_dict

    # Problem metadata
    prob_info = {
        'name':       cfg.problem.prob_name,
        'type':       cfg.problem.prob_type,
        'goal':       cfg.problem.opt_goal,
        'dimensions': cfg.problem.dimensions,
        'opt_global': cfg.problem.opt_global,
        'mean_value': cfg.problem.mean_value,
        'mean_weight': cfg.problem.mean_weight,
        'PID':        f"{cfg.problem.prob_name}_{cfg.problem.dimensions}"
    }

    # Instantiate fitness
    fitness_fn = getattr(FitnessFunctions, cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Check if algo uses dynamically determined mut rate and if so calculate value
    if "use_dynamic_mutation" in cfg.algo:
        if cfg.algo.use_dynamic_mutation:
            cfg.algo.init_args.mutate_params.indpb = call(cfg.algo.indpb_fn)
        else:
            cfg.algo.init_args.mutate_params.indpb = cfg.algo.static_indpb
    
    noise_val = cfg.problem.fitness_params.noise_intensity
    if cfg.run.get("use_noise_dependent_eval_limit", False):
        mapping = { k: int(v) for k, v in cfg.run.eval_limit_for_noise.items() }
        cfg.run.eval_limit = mapping.get(f"{noise_val}", cfg.run.eval_limit)

    # Algorithm class and params
    true_fit_params = fit_params.copy()
    true_fit_params['noise_intensity'] = 0

    algo_params = {
        'sol_length':            cfg.problem.dimensions,
        'opt_weights':           tuple(cfg.problem.weights),
        'eval_limit':            cfg.run.eval_limit,
        'attr_function':         getattr(Algorithms, cfg.problem.attr_function),
        'starting_solution':     None,
        'target_stop':           cfg.problem.opt_global,
        'gen_limit':             None,
        'fitness_function':      (fitness_fn, fit_params),
        'true_fitness_function': (fitness_fn, true_fit_params)
    }

    # Run and log via MLflow
    with mlflow.start_run(run_name=cfg.algo.name):
        # Log parameters
        mlflow.log_params({
            # 'pop_size':    cfg.algo.pop_size,
            # 'select_size': cfg.algo.select_size,
            'dimensions':  cfg.problem.dimensions,
            'seed':        cfg.run.seed,
            'max_gens':    cfg.run.max_gens,
            **{f"fit_{k}": v for k, v in fit_params.items()}
        })

        # Execute experiment (single or multirun seed)
        df = hydra_algo_data_multi(
            prob_info,
            cfg.algo.init_args,
            algo_params,
            num_runs=cfg.run.num_runs,
            base_seed=cfg.run.seed,
            parallel=cfg.run.parallel
        )

        # Log metrics and artifacts
        for row in df.itertuples():
            mlflow.log_metric('final_fitness', row.final_fit, step=row.seed)
        df.to_csv('results.csv', index=False)
        mlflow.log_artifact('results.csv')
        mlflow.log_artifact("outputs/.hydra/config.yaml")
    
    mlflow.end_run(status="FINISHED")

    # Print summary
    print(df[['seed', 'final_fit']])

if __name__ == '__main__':
    main()