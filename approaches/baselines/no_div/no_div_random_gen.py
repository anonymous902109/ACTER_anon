from src.approaches.backward_cfs.backward_generator import BackGen
from src.optimization.objs.back_obj import BackObj
from src.optimization.search.evolutionary.evolutionary_search_RAND import EvolutionSearchRAND


class NoDivRANDCFGen(BackGen):

    def __init__(self, env, bb_model, params=None):

        super(NoDivRANDCFGen, self).__init__(env, bb_model, params)
        self.obj = BackObj(env, bb_model, params)
        self.optim = EvolutionSearchRAND(env, bb_model, self.obj, params)
