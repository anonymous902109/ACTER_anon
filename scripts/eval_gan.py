import json

from src.approaches.baselines.cf.ganterfactual import GANterfactual
from src.envs.highway_env import HighwayEnv
from src.models.dqn_model import DQNModel
from src.utils.user_study_util import choose_user_study_traj
from src.utils.utils import generate_paths_with_outcome
import matplotlib.pyplot as plt


def main(task_name):
    if task_name == 'highway':
        env = HighwayEnv()

    training_timesteps = int(2e5)

    # define paths
    model_path = '../trained_models/{}'.format(task_name)
    failure_traj_path = '../datasets/{}/failures'.format(task_name)
    param_file = '../params/{}.json'.format(task_name)

    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # define bb model
    bb_model = DQNModel(env, model_path, training_timesteps)

    # extract trajectories that end in failure
    failure_trajectories = generate_paths_with_outcome(failure_traj_path, env, bb_model)

    # extract trajectories where failure can be prevented by changing only one action
    fail_success_pair = choose_user_study_traj(failure_trajectories, env,
                                               path='../datasets/{}/fail_success_pairs.pkl'.format(task_name))

    gan_cf = GANterfactual(env, bb_model, params)

    for i, (t, s) in enumerate(fail_success_pair):
        fact = t.states[-1]
        cf = gan_cf.generate_counterfactuals(t.states[-1], target=0)

        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        # plot fact and cf
        axes[0].imshow(fact[-1, ...].T, cmap=plt.get_cmap('gray'))
        axes[1].imshow(cf[-1, ...].T, cmap=plt.get_cmap('gray'))






if __name__ == '__main__':
    main('highway')