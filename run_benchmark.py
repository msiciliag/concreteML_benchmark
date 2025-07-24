import yaml
import click
import os
import random
from multiprocessing import Pool

from utils import load_progress, save_progress, hash_config, generate_experiment_configs
from experiment import run_experiment_iteration  # Assuming this is in the same directory

# Default values
DEFAULT_RANDOM_SEED = 42
DEFAULT_EXP_NUMBER = 20

def conduct_experiments(task_config: dict, concreteml_config: dict, model_config: dict, progress: set, progress_file: str, config_file: str, mode: str, exp_number: int):
    """Generates all configurations and runs the experiments."""
    configs_generator = generate_experiment_configs(task_config, concreteml_config, model_config, mode, exp_number)
    
    config_name = os.path.splitext(os.path.basename(config_file))[0]
    
    successful_runs = 0
    
    # For basic mode, we just need to iterate through all generated configs
    # For random mode, we need to ensure we complete exp_number of them
    target_runs = exp_number if mode == 'random' else -1 # -1 means iterate all

    while True:
        if mode == 'random' and successful_runs >= target_runs:
            break
        
        try:
            config = next(configs_generator)
        except StopIteration:
            break # No more configs to process

        print("-----------------------------------------")
        print(f"\nRunning experiment with configuration: {config}\n")
        
        config_hash = hash_config(config)
        if config_hash in progress:
            print(f"Skipping already tested configuration: {config}")
            successful_runs += 1 # Count skipped as successful for random mode progress
            continue
        
        successful_run = False
        while not successful_run:
            successful_run = run_experiment_iteration(config, task_config, model_config, concreteml_config, config_name)
            if not successful_run:
                print(f"Retrying configuration due to dataset error: {config}")

        progress.add(config_hash)
        save_progress(progress, progress_file)
        successful_runs += 1

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('progress_file', type=click.Path(), required=False, default=None)
@click.option('--mode', type=click.Choice(['basic', 'random']), default='basic', help='Execution mode: basic (grid search) or random (randomized search).')
@click.option('--exp-number', type=int, default=DEFAULT_EXP_NUMBER, help='Number of experiments for random mode.')
@click.option('--random-seed', type=int, default=DEFAULT_RANDOM_SEED, help='Random seed for random mode.')
@click.option('--clear-progress', is_flag=True, help='Clear progress and start from scratch.')
def main(config_file, progress_file, mode, exp_number, random_seed, clear_progress):
    """Main function to load configurations and run experiments."""
    if clear_progress and progress_file and os.path.exists(progress_file):
        os.remove(progress_file)

    with open(config_file) as f:
        full_config = yaml.safe_load(f)

    if progress_file is None:
        model_name = full_config["models"][0]["model"].get('name', 'default')
        progress_file = f"{model_name}_progress_{mode}.bin"

    progress = load_progress(progress_file)
    random.seed(random_seed)
    
    # For now, this runs models sequentially. Pool could be used for multiple model configs in one file.
    for model_group in full_config["models"]:
        conduct_experiments(full_config["task"], full_config["concreteml"], model_group["model"], progress, progress_file, config_file, mode, exp_number)

if __name__ == '__main__':
    main()