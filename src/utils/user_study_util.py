import copy
import pickle
import random
import pandas as pd
import re
import os

import matplotlib.pyplot as plt
from tqdm import tqdm


def choose_random_subset(traj, n=10):
    return random.sample(traj, n)


def choose_user_study_traj(traj, env, path, n=20):
    ''' Selects a dataset of trajectories where there exists at least one one-step prevention of the failure
    :param traj: dataset of all failure trajectories
    :param env: gym gym_env
    :param path: save path for trajectories
    :param n: number of trajectories to be selected
    '''
    try:
        with open(path, 'rb') as f:
            user_study_traj, test_traj = pickle.load(f)
            i=0
            for t in user_study_traj:
                t.id = i
                i += 1
            for t in test_traj:
                t.id = i
                i += 1

            print('Trajectores = {}'.format(len(user_study_traj)))
            print('Test trajectories = {}'.format(len(test_traj)))
    except FileNotFoundError:
        print('Choosing trajectories that can be fixed in one step from {} failure trajectories'.format(len(traj)))
        user_study_traj = []
        test_traj = []
        for j, t in tqdm(enumerate(traj)):
            unique_sols = 0
            # generate possible action sequences
            one_action_changed_seq = change_one_action(t.actions, env)
            for a_seq in one_action_changed_seq:
                env.reset()
                env.set_stochastic_state(copy.copy(t.states[0]), copy.deepcopy(t.env_states[0]))
                for a in a_seq:
                    obs, rew, done, trunc, info = env.step(a)

                if not env.check_failure() and not done:
                    print(a_seq)
                    unique_sols += 1
                    if unique_sols >= 2:
                        user_study_traj.append((t))
                        break

            if unique_sols == 1:
                test_traj.append(t)

        with open(path, 'wb') as f:
            pickle.dump((user_study_traj, test_traj), f)

    return user_study_traj, test_traj


def generate_correct_action_seq(env, t):
    corr = None
    unique_sols = 0

    one_action_changed_seq = change_one_action(t.actions, env)
    for a_seq in one_action_changed_seq:
        env.reset()
        env.set_stochastic_state(copy.copy(t.states[0]), copy.deepcopy(t.env_states[0]))
        for a in a_seq:
            obs, rew, done, trunc, info = env.step(a)

        if not env.check_failure() and not done:
            corr = a_seq
            unique_sols += 1
            if unique_sols >= 2:
                break

    return corr

def change_one_action(actions, env):
    action_seq = []
    i = 0

    for action_id in range(len(actions)):
        for alt_action in range(env.action_space.n):
            new_seq = copy.copy(actions)
            new_seq[action_id] = alt_action
            action_seq.append(new_seq)

    return action_seq

def save_frames_as_gif(frames, path, filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    # patch = plt.imshow(frames[0])
    # plt.axis('off')
    #
    # def animate(i):
    #     patch.set_data(frames[i])

    # anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    # anim.save(path + filename, writer='imagemagick', fps=60)

    for i, f in enumerate(frames):
        plt.imsave("{}/{}_{}.jpg".format(path, i, filename), f)


def render_traj_pairs(traj_pairs, env):
    for i, (f, s) in enumerate(traj_pairs):
        render_traj(f.env_states[0], f.states[0], f.actions, env, 'fail')
        render_traj(f.env_states[0], f.states[0], s.actions, env, 'success')


def render_traj(start, start_state, actions, env, path, file_name):
    env.reset()
    frames = []
    env.set_stochastic_state(start_state, copy.deepcopy(start))
    obs = start_state

    for a in actions:
        # calculating additional features to display
        speed = round(env.gym_env.road.vehicles[0].speed)
        print('Speed = {}'.format(speed))

        frames.append(env.render())
        obs, _, _, _, info = env.step(a)
        print('Action = {}'.format(a))

    speed = round(env.gym_env.road.vehicles[0].speed)
    print('Speed = {}'.format(speed))

    frames.append(env.render())
    save_frames_as_gif(frames, path=path, filename=file_name)


def generate_user_study_test_data(test_traj, traj, env, eval_path_results, render_path):
    print('--------- Rendering test data ---------')
    df = pd.read_csv(eval_path_results, header=0)

    ids = []
    for t in test_traj:
        corr_seq = generate_correct_action_seq(env, t)
        diff = [corr_seq[i] != t.actions[i] for i in range(len(corr_seq))]
        if diff[0] == 0 and diff[1] == 0:  # filter trajs that have less recent changes for simplifying user study
            ids.append(t.id)
            print('TEST: {} ACTIONS = {} CORRECT ACTIONS: {}'.format(t.id, t.actions, corr_seq))

    user_study_ids = random.sample(ids, 10)
    user_study_df = df[df['Fact id'].isin(user_study_ids)]

    render_user_study_traj(traj, env, render_path, user_study_df, user_study_ids)


def generate_user_study_train_data(trajs, env, eval_path_results, render_path, n=10):
    print('--------- Rendering training data ---------')
    df = pd.read_csv(eval_path_results, header=0)

    # select df where there is only one solution
    facts = list(df['Fact id'].values)
    unique_facts = [f for f in facts if facts.count(f) == 1]

    df = df[df['Fact id'].isin(unique_facts)]
    df = df[df['recency'] <= 0.2]
    unique_facts = list(df['Fact id'].values)

    # select random ones
    user_study_ids = random.sample(unique_facts, n)
    user_study_df = df[df['Fact id'].isin(user_study_ids)]

    render_user_study_traj(trajs, env, render_path, user_study_df, user_study_ids)


def render_user_study_traj(trajs, env, render_path, user_study_df, user_study_ids):
    # for each example generate a trajectory
    for i, fact_id in enumerate(user_study_ids):
        print('---------- ID = {} -----------'.format(fact_id))
        t = trajs[fact_id]
        start_env_state = t.env_states[0]
        start_state = t.states[0]

        print('------------- FACT -------------------')
        render_traj(start_env_state, start_state, t.actions, env, os.path.join(render_path, '{}'.format(i)), 'fact'.format(i))

        try:
            recourse = user_study_df[user_study_df['Fact id'] == fact_id]['Recourse'].values[0]
            recourse = [int(n) for n in re.findall('[0-9]', recourse)]
            print('------------- CF -------------------')
            render_traj(start_env_state, start_state, recourse, env, os.path.join(render_path, '{}'.format(i)),
                        'cf'.format(i))
        except IndexError: # if cf has not been generated
            return


