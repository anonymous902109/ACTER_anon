import json
import os

from src.approaches.backward_cfs.backward_generator import BackGen
from src.approaches.backward_cfs.backwards_generator_discrete import BackGenDiscrete
from src.approaches.explaining.cf_expl import CFExpl
from src.approaches.explaining.pf_expl import PFExpl
from src.approaches.raccer.fid_raccer import FidRACCER
from src.approaches.raccer.nsga_raccer import NSGARaccer
from src.envs.bike_sharing import BikeSharing
from src.envs.farm0 import Farm0
from src.envs.frozen_lake import FrozenLake
from src.envs.gridworld import Gridworld
from src.envs.highway_env import HighwayEnv
from src.evaluation.eval_diagnostic import generate_counterfactuals
from src.models.dqn_model import DQNModel
from src.outcomes.action_outcome import ActionOutcome
from src.outcomes.failure_outcome import FailureOutcome
from src.outcomes.one_action_outcome import OneActionOutcome
from src.utils.utils import seed_everything, load_facts_from_csv, generate_paths_with_outcome


def main(task_name):
    print('TASK = {} '.format(task_name))
    seed_everything(seed=1)

    # define paths
    model_path = '../trained_models/{}/{}'.format(task_name, task_name)
    param_file = '../params/{}.json'.format(task_name)
    outcome_traj_path = '../datasets/{}/facts/'.format(task_name)
    eval_path = f'../eval/{task_name}/'

    # define environment
    if task_name == 'highway':
        env = HighwayEnv()
        training_timesteps = int(2e5)
    elif task_name == 'farm0':
        env = Farm0()
        training_timesteps = int(1e5)
    elif task_name == 'gridworld':
        env = Gridworld()
        training_timesteps = int(2e5)
    elif task_name == 'frozen_lake':
        env = FrozenLake()
        training_timesteps = int(3e5)
    elif task_name == 'bikes':
        env = BikeSharing()
        training_timesteps = int(1e5)

    # load bb model
    bb_model = DQNModel(env, model_path, training_timesteps)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # define target outcomes
    failure_outcome = FailureOutcome(bb_model)
    one_action_outcomes = [OneActionOutcome(bb_model, target_action=a) for a in range(env.action_space.n)]

    outcomes = [one_action_outcomes[-1]]

    # generate facts
    facts = []
    for o in outcomes:
        f = generate_paths_with_outcome(o, os.path.join(outcome_traj_path, o.name), env, bb_model, horizon=params['horizon'])
        facts.append(f[0:10])

    # define algorithms for explaining
    pf = PFExpl(env, bb_model, params)
    cf = CFExpl(env, bb_model, params)

    methods = [pf, cf]
    method_names = ['Pf_expl', 'Cf_expl']

    for i, f in enumerate(facts):
        generate_counterfactuals(methods, method_names, f, outcomes[i], env, eval_path, params)


if __name__ == '__main__':
    tasks = ['gridworld']

    for t in tasks:
        main(t)
