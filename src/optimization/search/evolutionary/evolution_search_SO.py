from src.optimization.algs.evolutionary.evol_alg_SO import EvolutionAlgSO
from src.optimization.search.abstract_search import AbstractSearch


class EvolutionSearchSO(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        super(EvolutionSearchSO, self).__init__(env, bb_model, obj, params)
        self.alg = EvolutionAlgSO(env, bb_model, obj, params)





