from src.approaches.backward_cfs.backward_generator import BackGen
from src.optimization.objs.semifact_obj import SemifactObj
from src.optimization.search.evolutionary_search_SO import EvolutionSearchRAND


class NoDivSFGen(BackGen):

    def __init__(self, env, bb_model, params=None):
        self.obj = SemifactObj(env, bb_model, params)
        self.optim = EvolutionSearchRAND(env, bb_model, self.obj, params)

        super(NoDivSFGen, self).__init__(env, bb_model, params)