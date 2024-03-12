import copy
import json
import os
import random
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from pfrl.replay_buffers import EpisodicReplayBuffer
from tqdm import tqdm

from src.models.trajectory import Trajectory
from src.utils.highligh_div import HighlightDiv


def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def load_facts_from_summary(env, bb_model, num_states=10):
    highlight = HighlightDiv(env, bb_model, num_states=30)
    facts = highlight.generate_important_states()

    return facts, [[4, 5]] * len(facts)

def generate_summary_states(env, bb_model, df):
    states = df['fact'].values
    trans_states = []
    ind = 0
    indices = []

    for s in states:

        s = s.strip('][').split(', ')
        s = [int(r) for r in s]
        if s not in trans_states:
            trans_states.append(s)
            indices.append(ind)
        ind += 1

    trans_states = np.array(trans_states)

    highlight = HighlightDiv(env, bb_model, num_states=20)
    summary_state_ind = highlight.select_important_states(trans_states, indices)

    df.reset_index(inplace=True)
    return df[df.index.isin(summary_state_ind)]

def load_facts_from_json(fact_file):
    with open(fact_file, 'r') as f:
        content = json.load(f)

    facts = []
    targets = []
    for fact in content:
        facts.append(fact)
        target = fact['target']
        targets.append([target])

    return facts, targets


def generate_paths_with_outcome(outcome, csv_path, env, bb_model, n_ep=1000, horizon=5):
    ''' Generates a dataset of Trajectory objects where a failure happens
    :param csv_path: path to save the dataset
    :param env: gym gym_env
    :param bb_model: a model used for prediciting next action in the gym_env
    :param n_ep: number of episodes
    :param horizon: number of iteractions before the failure that are saved in the failure trajectory
    '''
    try:
        # load buffer
        buffer = EpisodicReplayBuffer(capacity=100000000)
        buffer.load(csv_path)
        episodes = buffer.sample_episodes(n_episodes=len(buffer.episodic_memory))
        # transform into traj class
        trajs = []
        episodes = [e for e in episodes if len(e) >= horizon]
        return combine_trajs(episodes, outcome)

    except FileNotFoundError:
        print('Generating failure trajectories...')
        buffer = EpisodicReplayBuffer(capacity=10000000)

        for i in tqdm(range(n_ep)):
            obs, _ = env.reset(int(datetime.now().timestamp()))
            done = False
            p = []
            while not done:
                action = bb_model.predict(obs)
                p.append((copy.copy(obs), action, None, None, copy.deepcopy(env.get_env_state())))

                new_obs, rew, done, trunc, info = env.step(action)
                done = done or trunc

                if outcome.explain_outcome(env, new_obs):
                    p.append((copy.copy(new_obs), None, None, None, copy.deepcopy(env.get_env_state()))) # add the last state -- failure state with None as action identifier
                    if (len(p) - 1) >= horizon:  # have to subtract the last state because it doesn't have an action with it
                        for t in p[-(horizon+1):]:
                            buffer.append(*t)

                        buffer.stop_current_episode()

                    done = True

                obs = new_obs

        # save buffer
        buffer.save(csv_path)
        episodes = buffer.sample_episodes(n_episodes=len(buffer.episodic_memory))

        return combine_trajs(episodes, outcome)


def combine_trajs(episodes, outcome):
    # transform into traj class
    trajs = []
    for e_id, e in enumerate(episodes):
        t = Trajectory(e_id, outcome)
        for i in e:
            t.append(i['state'], i['action'], i['next_action'])

        t.outcome.true_action = t.actions[-1]
        trajs.append(t)

    print('Generated {} failure trajectories'.format(len(trajs)))
    return trajs


def load_facts_from_csv(csv_path, env, bb_model, n_ep=100):
    try:
        df = pd.read_csv(csv_path, header=0)
        facts = df.values
        return facts, None

    except FileNotFoundError:
        print('Generating facts...')
        data = []

        for i in range(n_ep):
            obs = env.reset()
            done = False

            while not done:
                data += [list(obs)]

                choice = random.randint(0, 1)
                action = bb_model.predict(obs) if choice == 0 else env.action_space.sample()

                obs, rew, done, _ = env.step(action)

        dataframe = pd.DataFrame(data)
        dataframe.drop_duplicates(inplace=True)

        dataframe = dataframe.sample(100)

        dataframe.to_csv(csv_path, index=None)

        return dataframe.values, None