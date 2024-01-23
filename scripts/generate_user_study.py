import json
import os
from random import random

import pandas as pd
from src.envs.farm0 import Farm0
from src.envs.highway_env import HighwayEnv

from src.models.dqn_model import DQNModel
from src.utils.user_study_util import choose_user_study_traj, choose_random_subset, render_traj, \
    generate_user_study_train_data, generate_user_study_test_data
from src.utils.utils import seed_everything, generate_unsuccessful_paths


def main(task_name):
    print('Generating user study states using {} environment '.format(task_name))
    seed_everything(seed=1)

    # define paths
    param_file = '../params/{}.json'.format(task_name)
    model_path = '../trained_models/{}'.format(task_name)
    failure_traj_path = '../datasets/{}/failures'.format(task_name)
    one_step_traj_path = '../datasets/{}/fail_success_pairs.pkl'.format(task_name)
    eval_path = f'../eval/{task_name}/'
    user_study_train_path = '../eval/{}/user_study_train/'.format(task_name)
    user_study_test_path = '../eval/{}/user_study_test/'.format(task_name)

    # define environment
    if task_name == 'highway':
        env = HighwayEnv()
        training_timesteps = int(2e5)
    elif task_name == 'farm0':
        env = Farm0()
        training_timesteps = int(2e5)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # define bb model
    bb_model = DQNModel(env, model_path, training_timesteps)

    # extract trajectories that end in failure
    failure_trajectories = generate_unsuccessful_paths(failure_traj_path, env, bb_model, horizon=params['horizon'])

    # get test trajectories that have a solution in one step
    failure_trajectories, test_traj = choose_user_study_traj(failure_trajectories, env, one_step_traj_path)

    user_study_train = choose_random_subset(failure_trajectories, n=10)
    user_study_test = choose_random_subset(test_traj, n=10)

    # define algorithms
    method_names = ['MOO_CF']

    for m in method_names:
        eval_path_results = os.path.join(eval_path, f'{m}/results_{5}.csv')

        generate_user_study_train_data(failure_trajectories + test_traj, env, eval_path_results, user_study_train_path)
        generate_user_study_test_data(test_traj, failure_trajectories + test_traj, env, eval_path_results, user_study_test_path)


if __name__ == '__main__':
    main('highway')
