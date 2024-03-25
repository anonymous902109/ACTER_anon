from src.approaches.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.diagnosing.cf_diag_obj import CfDiagObj
from src.optimization.objs.explaining.cf_expl_obj import CfExplObj
from src.optimization.search.evolutionary.evolution_search_MO import EvolutionSearchMOO


class CFExpl(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = CfDiagObj(env, bb_model, params)
        self.optim = EvolutionSearchMOO(env, bb_model, self.obj, params)

        super(CFExpl, self).__init__()

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        res = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        cfs = []

        for cf in res:
            cfs.append(CF(fact, cf[0], None, cf[2], cf[1]))

        return cfs
