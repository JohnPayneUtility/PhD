import random
import pickle
import pandas as pd
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig
import mlflow
from RunManager import algo_data_multi
import Algorithms, FitnessFunctions, ExperimentsHelpers
# from run_helpers import dynamic_pop_size_UMDA

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialise MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Determine population size
    if cfg.algo.use_dynamic_pop_size:
        pop_size = call(cfg.algo.pop_size_fn)
    else:
        pop_size = cfg.algo.static_pop_size
    cfg.algo.pop_size = pop_size

    # Set random seed
    random.seed(cfg.run.seed)

    # Problem metadata
    prob_info = {
        'name':       cfg.problem.prob_name,
        'type':       cfg.problem.prob_type,
        'goal':       cfg.problem.opt_goal,
        'dimensions': cfg.problem.dimensions,
        'opt_global': cfg.problem.opt_global,
        'mean_value': cfg.problem.mean_value,
        'mean_weight': cfg.problem.mean_weight,
        'PID':        f"OneMax_{cfg.problem.dimensions}"
    }

    # Instantiate fitness
    fitness_fn = getattr(FitnessFunctions, cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Algorithm class and params
    algo_class = getattr(Algorithms, cfg.algo.type)
    algo_params = {
        'pop_size':              cfg.algo.pop_size,
        'select_size':           cfg.algo.select_size,
        'sol_length':            cfg.problem.dimensions,
        'opt_weights':           tuple(cfg.problem.weights),
        'eval_limit':            cfg.run.eval_limit,
        'attr_function':         getattr(Algorithms, cfg.problem.attr_function),
        'starting_solution':     None,
        'target_stop':           cfg.problem.opt_global,
        'gen_limit':             None,
        'fitness_function':      (fitness_fn, fit_params),
        'true_fitness_function': (fitness_fn, {'noise_intensity': 0})
    }

    # Run and log via MLflow
    with mlflow.start_run(run_name=cfg.algo.name):
        # Log parameters
        mlflow.log_params({
            'pop_size':    cfg.algo.pop_size,
            'select_size': cfg.algo.select_size,
            'dimensions':  cfg.problem.dimensions,
            'seed':        cfg.run.seed,
            'max_gens':    cfg.run.max_gens,
            **{f"fit_{k}": v for k, v in fit_params.items()}
        })

        # Execute experiment (single or multirun seed)
        df = algo_data_multi(
            prob_info,
            algo_class,
            algo_params,
            num_runs=cfg.run.num_runs,
            base_seed=cfg.run.seed,
            parallel=False
        )

        # Log metrics and artifacts
        for row in df.itertuples():
            mlflow.log_metric('final_fitness', row.final_fit, step=row.seed)
        df.to_csv('results.csv', index=False)
        mlflow.log_artifact('results.csv')
    
    mlflow.end_run(status="FINISHED")

    # Print summary
    print(df[['seed', 'final_fit']])

if __name__ == '__main__':
    main()