import copy

from src.envs.farm0 import Farm0
import numpy as np

from src.models.dqn_model import DQNModel


def main():
    env = Farm0()
    # define bb model
    model_path = '../trained_models/{}'.format('farm0')
    bb_model = DQNModel(env, model_path, 0)

    np_gen = np.random.default_rng(0)
    # env.gym_env.np_random = np.random.default_rng(5)
    obs, _ = env.reset(5)
    env.gym_env.fields['Field-0'].entities['Plant-0'].set_random(np.random.default_rng(10))
    # env.gym_env.fields['Field-0'].entities['Soil-0'].set_random(np.random.default_rng(5))
    # env.gym_env.fields['Field-0'].entities['Weather-0'].set_random(np.random.default_rng(10))

    i = 0
    done = False
    while not done:
        i += 1
        action = bb_model.predict(obs)
        obs, _, done, _, _ = env.step(action)

    print(env.gym_env.fields['Field-0'].entities['Plant-0'].variables)
    print('-------------------------------------------------')

    # env.gym_env.np_random = np.random.default_rng(5)
    obs, _ = env.reset(5)
    env.gym_env.fields['Field-0'].entities['Plant-0'].set_random(np.random.default_rng(10))
    # env.gym_env.fields['Field-0'].entities['Soil-0'].set_random(np.random.default_rng(5))
    # env.gym_env.fields['Field-0'].entities['Weather-0'].set_random(np.random.default_rng(10))

    i = 0
    done = False
    while not done:
        i += 1
        action = bb_model.predict(obs)
        obs, _, done, _, _ = env.step(action)

    print(env.gym_env.fields['Field-0'].entities['Plant-0'].variables)

if __name__ == '__main__':
    main()