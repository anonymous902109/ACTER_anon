import copy
from datetime import datetime

import numpy as np

from src.optimization.objs.abs_obj import AbstractObj


class BackObj(AbstractObj):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for discrete actions
    '''

    def __init__(self, env, bb_model, params):

        super(BackObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'proximity', 'sparsity', 'recency']
        self.constraints = ['validity']

        self.n_sim = params['n_sim']

    def get_objectives(self, fact, cf, actions, target_action):
        proximity = self.action_proximity(fact.actions, actions)
        sparsity = self.sparsity(fact, actions)
        recency = self.recency(fact, actions)
        stochasticity = self.stoch_validity(fact, actions)

        return {'uncertainty': stochasticity,
                'proximity': proximity,
                'sparsity': sparsity,
                'recency': recency}

    def get_constraints(self, fact, cf, actions, target_action):
        validity = self.validity(fact, actions)

        return {'validity': validity}

    def validity(self, fact, actions):
        self.env.reset()
        self.env.set_stochastic_state(copy.copy(fact.states[0]), copy.deepcopy(fact.env_states[0]))
        for a in actions:
            _, _, done, trunc, _ = self.env.step(a)
            if done or trunc or self.env.check_failure():
                break

        # IMPORTANT: return 1 if the class hasn't changed -- to be compatible with minimization used by NSGA
        return self.env.check_failure()

    def sparsity(self, fact, actions):
        return 1 - (sum(np.array(fact.actions) == np.array(actions)) / len(actions))

    def recency(self, fact, actions):
        diff = [fact.actions[i] != actions[i] for i in range(len(actions))]

        n = len(actions)
        k = 2.0/(n * (n + 1))
        weights = [k * (i+1) for i in range(len(actions))]

        weights.reverse() # the biggest penalty for the first (least recent) action

        recency = sum([diff[i] * weights[i] for i in range(len(actions))])

        return recency

    def stoch_validity(self, fact, actions):
        n_sim = self.n_sim
        cnt = 0
        for i in range(n_sim):
            randomseed = int(datetime.now().timestamp())
            self.env.reset(seed=randomseed)
            self.env.set_nonstoch_state(copy.deepcopy(fact.states[0]), copy.deepcopy(fact.env_states[0]))
            for a in actions:
                obs, rew, done, trunc, _ = self.env.step(a)
                if done or trunc or self.env.check_failure():
                    break

            if not self.env.check_failure():
                cnt += 1

        return 1 - ((cnt * 1.0)/n_sim)

    def action_proximity(self, fact, actions):
        dist = 0
        for i, a in enumerate(actions):
            dist += self.env.action_distance(a, fact[i])

        avg_distance = dist / (1.0*len(actions))
        return avg_distance

    def bool_dist(self, x, y):
        return x != y

    def euclid_dist(self, x, y):
        return abs(x - y)
