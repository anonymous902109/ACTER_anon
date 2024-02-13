import copy
import random

import numpy as np

from src.optimization.algs.causal.state_importance_alg import StateImportanceAlg


class InterestingnessAlgNoFreq(StateImportanceAlg):

    def __init__(self, env, bb_model, obj, params):
        super(InterestingnessAlgNoFreq, self).__init__(env, bb_model, obj, params)

    def get_interesting_state(self, fact):
        freq = []
        for x in fact.states:
            freq.append(self.freq[x])

        max_freq = np.argmin(freq)

        return max_freq














