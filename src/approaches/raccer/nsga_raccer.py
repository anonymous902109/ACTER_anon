from src.approaches.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.fid_obj import FidObj
from src.optimization.search.evolutionary.evolution_search_MO import EvolutionSearchMOO
from src.optimization.search.tree.heuristic_tree_search import HeuristicTreeSearch


import numpy as np


class NSGARaccer(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = FidObj(env, bb_model, params)
        self.optim = EvolutionSearchMOO(env, bb_model, self.obj, params)

        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']

        super(NSGARaccer, self).__init__()

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        res = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        cfs = []

        for cf in res:
            cfs.append(CF(fact, cf[0], None, cf[2], cf[1]))

        return cfs
