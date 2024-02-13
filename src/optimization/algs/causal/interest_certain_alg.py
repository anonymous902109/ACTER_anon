import copy
import math
import random

import numpy as np

from src.optimization.algs.causal.state_importance_alg import StateImportanceAlg


class InterestingnessAlgCertain(StateImportanceAlg):

    def __init__(self, env, bb_model, obj, params):
        super(InterestingnessAlgCertain, self).__init__(env, bb_model, obj, params)

    def get_interesting_state(self, fact):
        certainty = []
        for x in fact.states:
            actions = self.env.get_actions(x)
            prob = [self.bb_model.get_action_prob(x, a) * math.log(self.bb_model.get_action_prob(x, a), math.e) for a in actions]

            # High evenness indicates similar values of Q values -- indicating uncertainty
            evenness = -sum([p / math.log(len(prob), math.e) for p in prob])

            certainty.append(evenness)

        min_certainty = np.argmin(certainty)

        return min_certainty














