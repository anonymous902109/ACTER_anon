import copy
import random

import numpy as np

from src.optimization.algs.causal.state_importance_alg import StateImportanceAlg


class HighlightsAlg(StateImportanceAlg):

    def __init__(self, env, bb_model, obj, params):
        super(HighlightsAlg, self).__init__(env, bb_model, obj, params)

    def get_interesting_state(self, fact):
        Q_diffs = []
        for x in fact.states:
            Q_vals = self.bb_model.get_Q_vals(x)
            diff = abs(max(Q_vals) - min(Q_vals))
            Q_diffs.append(diff)

        cause_id = np.argmax(Q_diffs)

        return cause_id