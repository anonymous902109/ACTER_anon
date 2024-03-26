import copy
import math
import random
from datetime import datetime

import numpy as np

import torch

from src.approaches.models.enc_dec import EncoderDecoder


class AbstractObjective:
    ''' Describes an objective function for counterfactual search '''

    def __init__(self, env, bb_model, params):
        self.env = env
        self.bb_model = bb_model

        self.n_sim = params['n_sim']
        self.max_actions = params['max_actions']

        self.noop = -1

        self.enc_dec = EncoderDecoder(self.env, self.bb_model, path='../datasets/{}/'.format(params['task_name']), k=params['horizon'])

    def get_first_state(self, fact):
        return None, None

    def action_proximity(self, fact, actions):
        fact_traj = self.combine(fact.states, fact.actions)
        cf_traj = self.get_trajectory(fact, actions)

        fact_enc = self.enc_dec.encode(fact_traj)
        cf_enc = self.enc_dec.encode(cf_traj)

        distance = math.sqrt(sum((fact_enc - cf_enc) ** 2))

        return distance

    def validity(self, fact, actions):
        self.env.reset()
        first_state = self.get_first_state(fact)
        self.env.set_stochastic_state(*first_state)

        early_break = False
        if len(actions) == 0:
            return 1

        for a in actions:
            obs, _, done, trunc, _ = self.env.step(a)
            if done or trunc or self.env.check_failure():
                early_break = True
                break

        if early_break:
            return 1

        valid_outcome = fact.outcome.cf_outcome(self.env, obs)
        # IMPORTANT: return 1 if the class hasn't changed -- to be compatible with minimization used by NSGA
        return not valid_outcome

    def sparsity(self, fact, actions):
        return 1 - (sum(np.array(fact.actions) == np.array(actions)) / len(actions))

    def recency(self, fact, actions):
        diff = [fact.actions[i] != actions[i] for i in range(len(actions))]

        n = len(actions)
        k = 2.0/(n * (n + 1))
        weights = [k * (i+1) for i in range(len(actions))]

        weights.reverse()  # the biggest penalty for the first (least recent) action

        recency = sum([diff[i] * weights[i] for i in range(len(actions))])

        return recency

    def reachability(self, actions):

        if len(actions) == 0:
            return 1

        return len(actions) * 1.0 / self.max_actions

    def stoch_validity(self, fact, actions):
        n_sim = self.n_sim
        cnt = 0.0
        for i in range(n_sim):
            randomseed = int(datetime.now().timestamp())
            self.env.reset(randomseed)
            first_state = self.get_first_state(fact)
            self.env.set_nonstoch_state(*first_state)

            if len(actions) == 0:
                break

            early_break = False
            for a in actions:
                obs, rew, done, trunc, _ = self.env.step(a)
                if done or trunc or self.env.check_failure():
                    early_break = True
                    break

            valid_outcome = fact.outcome.cf_outcome(self.env, copy.deepcopy(obs))
            if valid_outcome and (not early_break):
                cnt += 1

        return 1 - (cnt/n_sim)

    def combine(self, states, actions):
        ''' Combines a list of states and actions into format (s1, ..., sn, a1, ... , an) '''
        comb = []
        for s in states:
            # all all states first
            comb.extend(list(s.flatten()))

        for a in actions:
            # add all actions
            comb.append(a)

        return torch.tensor(comb)

    def get_trajectory(self, fact, actions):
        self.env.reset()

        first_state = self.get_first_state(fact)
        self.env.set_stochastic_state(*first_state)

        t = []
        t.extend(list(fact.states[0].flatten()))
        i = 0
        for a in actions:
            obs, _, done, trunc, _ = self.env.step(a)

            t.extend(list(obs.flatten()))  # add the last state too

            i += 1
            if done or trunc or self.env.check_failure():
                break

        while i < len(actions):
            # not all actions have been executed because of validity being broken
            t.extend(list(self.env.reset()[0].flatten()))  # add a random state to fill up space
            i += 1

        for a in actions:
            t.append(a)

        return torch.tensor(t)

    def fidelity(self, fact, actions, bb_model):
        # run simulations from fact with actions
        n_sim = self.n_sim

        fidelities = []

        for s in range(n_sim):
            self.env.reset()
            # set env to last state -- failure state, this is used by raccer
            first_state = self.get_first_state(fact)
            self.env.set_stochastic_state(*first_state)

            obs = fact.end_state

            fid = 1.0

            if len(actions) == 0:
                return 1

            done = False
            early_break = False
            available_actions = self.env.get_actions(fact)
            ep_rew = 0.0
            for a in actions:
                if done or (a not in available_actions) or (len(available_actions) == 0):
                    early_break = True
                    break

                prob = bb_model.get_action_prob(obs, a)
                fid *= prob

                obs, rew, done, trunc, _ = self.env.step(a)
                ep_rew += rew

                available_actions = self.env.get_actions(obs)

            if not early_break:
                fidelities.append(fid)

        if len(fidelities):
            fidelity = sum(fidelities) / (len(fidelities) * 1.0)
        else:
            fidelity = 0

        # IMPORTANT -- returning False when fidelity is satisfied to be compatible with NSGA-II minimizing objective
        return fidelity < 0.1

    def calculate_end_state(self, state, env_state, actions):
        self.env.reset()
        self.env.set_stochastic_state(copy.deepcopy(state), copy.deepcopy(env_state))
        for a in actions:
            obs, _, done, trunc, _ = self.env.step(a)

        return obs