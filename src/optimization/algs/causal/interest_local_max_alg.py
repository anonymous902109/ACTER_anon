import copy
import random

import numpy as np

from src.optimization.algs.causal.state_importance_alg import StateImportanceAlg


class InterestingnessAlgLocalMax(StateImportanceAlg):

    def __init__(self, env, bb_model, obj, params):
        super(InterestingnessAlgLocalMax, self).__init__(env, bb_model, obj, params)

    def get_interesting_state(self, fact):
        for i, x in enumerate(fact.states):
            Q_x = self.bb_model.get_Q_vals(x)
            v_x = np.mean(Q_x)

            next_values = []
            for a in self.env.get_actions(x):
                self.env.reset()
                self.env.set_nonstoch_state(copy.copy(fact.states[0]), copy.deepcopy(fact.env_states[i]))
                next_s = self.env.step(a)[0]

                Q_next_s = self.bb_model.get_Q_vals(next_s)
                v_next_s = np.mean(Q_next_s)

                next_values.append(v_next_s)

            if v_x >= max(next_values):
                return i

        return None














