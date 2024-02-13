from src.optimization.algs.evolutionary.evol_alg_RAND import EvolutionAlgRAND
from src.optimization.search.abstract_search import AbstractSearch


class EvolutionSearchRAND(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = EvolutionAlgRAND(env, bb_model, obj, params)
        super(EvolutionSearchRAND, self).__init__(env, bb_model, obj, params)





