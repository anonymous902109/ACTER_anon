from src.approaches.backward_cfs.backward_generator import BackGen
from src.optimization.objs.back_obj import BackObj
from src.optimization.search.evolutionary.evolution_search_SO import EvolutionSearchSO


class NoDivSOCFGen(BackGen):

    def __init__(self, env, bb_model, params=None):
        super(NoDivSOCFGen, self).__init__(env, bb_model, params)
        self.obj = BackObj(env, bb_model, params)
        self.optim = EvolutionSearchSO(env, bb_model, self.obj, params)
