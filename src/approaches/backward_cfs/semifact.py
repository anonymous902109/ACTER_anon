from src.approaches.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.semifact_obj import SemifactObj
from src.optimization.search.evolution_search import EvolutionSearchMOO


class SemifactGen(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = SemifactObj(env, bb_model, params)
        self.optim = EvolutionSearchMOO(env, bb_model, self.obj, params)

        super(SemifactGen, self).__init__()

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        ''' Returns all cfs found in the tree '''
        res = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        cfs = []
        for cf in res:
            cfs.append(CF(cf[0], None, cf[2], cf[1]))

        return cfs