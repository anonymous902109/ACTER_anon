import copy
import pickle
import random

import numpy as np
from tqdm import tqdm


class StateImportanceAlg:

    def __init__(self, env, bb_model, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.params = params

        self.horizon = params['horizon']
        self.xu = params['xu']
        self.xl = params['xl']
        self.expl_budget = params['expl_budget']

    def search(self, init_state, fact, target_action):
        cause_id = self.get_interesting_state(fact)

        if cause_id is None:
            return []

        start_env_state = fact.env_states[0]
        start_state = fact.states[0]
        recourse = copy.deepcopy(fact.actions)
        alt_actions = self.choose_alt_action(start_state, start_env_state, recourse, cause_id, expl_budget=self.expl_budget)

        res = []
        for aa in alt_actions:
            recourse = copy.deepcopy(fact.actions)
            recourse[cause_id] = aa

            # evaluate recourse
            objectives = self.obj.get_objectives(fact, None, recourse, None)
            value = sum(objectives.values())
            constraints = self.obj.get_constraints(fact, None, recourse, None)

            satisfies = sum(list(constraints.values())) == 0  # 0 means it satisfies

            objectives.update(constraints)
            if satisfies:
                res.append((recourse, value, objectives))

        return res

    def get_interesting_state(self, fact):
        pass

    def choose_alt_action(self, start_state, start_env_state, fact, cause_id, expl_budget=5):
        # TODO: works for discrete actions only for now
        actions = np.arange(0, self.env.action_space.n)
        if len(actions) > expl_budget:
            actions = random.sample(list(actions), expl_budget)

        alt_actions = []
        for a in actions:
            try_seq = copy.deepcopy(fact)
            try_seq[cause_id] = a

            self.env.reset()
            self.env.set_nonstoch_state(copy.deepcopy(start_state), copy.deepcopy(start_env_state))

            for a in try_seq:
                self.env.step(a)

            if not self.env.check_failure():
                alt_actions.append(a)

        if len(alt_actions):
            return alt_actions

        # return a random action if none of them are valid
        return [self.env.action_space.sample()]
























