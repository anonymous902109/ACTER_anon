import json

from src.approaches.backward_cfs.backward_generator import BackGen
from src.approaches.backward_cfs.backwards_generator_discrete import BackGenDiscrete
from src.approaches.raccer.fid_raccer import FidRACCER
from src.envs.farm0 import Farm0
from src.envs.frozen_lake import FrozenLake
from src.envs.gridworld import Gridworld
from src.envs.highway_env import HighwayEnv
from src.evaluation.eval_diagnostic import generate_counterfactuals
from src.models.dqn_model import DQNModel
from src.utils.utils import seed_everything, load_facts_from_csv, generate_unsuccessful_paths


def main(task_name, agent_type):
    print('TASK = {} '.format(task_name))
    seed_everything(seed=1)

    # define paths
    model_path = 'trained_models/{}_{}'.format(task_name, agent_type)
    fact_csv_dataset_path = 'datasets/{}/facts.csv'.format(task_name, agent_type)
    fact_json_path = 'fact/{}.json'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)
    generator_path = 'trained_models/generator_{}_{}.ckpt'.format(task_name, agent_type)
    failure_traj_path = '../datasets/{}/failures'.format(task_name)
    eval_path = f'../eval/{task_name}/'

    # define environment
    if task_name == 'highway':
        env = HighwayEnv()
        training_timesteps = int(2e5)
    elif task_name == 'farm0':
        env = Farm0()
        training_timesteps = int(5e4)
    elif task_name == 'gridworld':
        env = Gridworld()
        gym_env = env
    elif task_name == 'frozen_lake':
        env = FrozenLake()
        gym_env = env

    # load bb model
    bb_model = DQNModel(gym_env, model_path, training_timesteps)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # load facts
    if task_name == 'frozen_lake' or task_name == 'gridworld':
        facts, targets = load_facts_from_csv(fact_csv_dataset_path, env, bb_model)
    elif task_name == 'highway' or task_name == 'farm0':
        facts = generate_unsuccessful_paths(failure_traj_path, env, bb_model, horizon=params['horizon'])

    # define algorithms
    acter = BackGen(env, bb_model, params)
    acter_discrete = BackGenDiscrete(env, bb_model, params)
    fid_raccer = FidRACCER(env, bb_model, params)

    methods = [acter_discrete, acter, fid_raccer]
    method_names = ['ACTER_discrete', 'ACTER', 'raccer']

    generate_counterfactuals(methods, method_names, facts, env, eval_path, params)


if __name__ == '__main__':
    tasks = ['frozen_lake', 'gridworld', 'farm0', 'highway']
    agent_types = ['optim', 'suboptim', 'non_optim']

    for t in tasks:
        for a in agent_types:
            main(t, a)
