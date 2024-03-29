from src.approaches.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.back_obj import BackObj
from src.optimization.search.causal.interest_freq_search import InterestFreqSearch


class InterestFreqGen(AbstractBaseline):

    def __init__(self, env, bb_model, params=None):
        self.obj = BackObj(env, bb_model, params)
        self.optim = InterestFreqSearch(env, bb_model, self.obj, params)

        super(InterestFreqGen, self).__init__()

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        res = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        cfs = []

        for cf in res:
            cfs.append(CF(cf[0], None, cf[2], cf[1]))

        return cfs