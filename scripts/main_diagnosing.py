import copy
import json
import sys

from src.approaches.backward_cfs.backward_generator import BackGen
from src.approaches.baselines.no_div.no_div_cf_gen import NoDivSOCFGen
from src.approaches.baselines.no_div.no_div_random_gen import NoDivRANDCFGen
from src.approaches.baselines.state_importance.highlights_cf import HIGHLIGHTS

from src.approaches.baselines.state_importance.interest_certain_gen import InterestCertainGen
from src.approaches.baselines.state_importance.interest_local_max_gen import InterestLocalMaxGen
from src.approaches.baselines.state_importance.interest_local_min_gen import InterestLocalMinGen
from src.approaches.baselines.state_importance.interest_uncertain_gen import InterestUncertainGen
from src.envs.farm0 import Farm0
from src.envs.highway_env import HighwayEnv
from src.evaluation.eval_diagnostic import generate_counterfactuals, evaluate_cf_properties, evaluate_diversity, \
    evaluate_coverage
from src.models.dqn_model import DQNModel
from src.utils.user_study_util import choose_user_study_traj, choose_random_subset
from src.utils.utils import seed_everything, generate_unsuccessful_paths


def main(task_name):
    print('TASK = {} '.format(task_name))
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
        training_timesteps = int(5e4)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # define bb model
    bb_model = DQNModel(env, model_path, training_timesteps)

    # extract trajectories that end in failure
    failure_trajectories = generate_unsuccessful_paths(failure_traj_path, env, bb_model, horizon=params['horizon'])

    # define algorithms
    backgen = BackGen(env, bb_model, params)  # our approach

    # state importance approaches
    state_importance_methods = []

    highlights = HIGHLIGHTS(env, bb_model, params)
    interest_certain = InterestCertainGen(env, bb_model, params)
    interest_uncertain = InterestUncertainGen(env, bb_model, params)
    interest_local_min = InterestLocalMinGen(env, bb_model, params)
    interest_local_max = InterestLocalMaxGen(env, bb_model, params)
    state_importance_methods += [highlights, interest_certain, interest_uncertain, interest_local_min,
                                 interest_local_max]

    # diversity baseline approaches
    diversity_methods = []
    for d in params['diversity_budget_list']:
        params['div_budget'] = d
        no_div_so_backgen = NoDivSOCFGen(env, bb_model, params)
        no_div_rand_backgen = NoDivRANDCFGen(env, bb_model, params)

        diversity_methods += [no_div_so_backgen, no_div_rand_backgen]

    # methods = [backgen] + state_importance_methods
    #
    # method_names = ['MOO_CF', 'HIGHLIGHTS', 'InterestCERTAIN', 'InterestUNCERTAIN', 'InterestLOCAL_MIN',
    #                 'InterestLOCAL_MAX', ]

    methods = diversity_methods
    method_names = ['NO_DIV_SO']

    # # generate counterfactuals
    generate_counterfactuals(methods, method_names, failure_trajectories, env, eval_path, params)

    # evaluate counterfactuals
    evaluate_cf_properties(methods, method_names, eval_path, params)
    evaluate_coverage(methods, method_names, eval_path, 247, params)
    evaluate_diversity(methods, method_names, eval_path, params)


if __name__ == '__main__':
    main('farm0')
    # main('highway')