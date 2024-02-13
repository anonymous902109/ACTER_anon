import copy
import random

import numpy as np

from src.optimization.algs.causal.state_importance_alg import StateImportanceAlg


class InterestingnessAlgFreq(StateImportanceAlg):

    def __init__(self, env, bb_model, obj, params):
        super(InterestingnessAlgFreq, self).__init__(env, bb_model, obj, params)

    def get_interesting_state(self, fact):
        f = []
        for x in fact.states:

            f.append(self.freq[tuple(x.flatten())])

        max_freq = np.argmax(f)

        return max_freq














