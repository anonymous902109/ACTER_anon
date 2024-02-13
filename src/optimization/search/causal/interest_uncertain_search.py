from src.optimization.algs.causal.interest_uncertain_alg import InterestingnessAlgUncertain
from src.optimization.search.abstract_search import AbstractSearch


class InterestUncertainSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgUncertain(env, bb_model, obj, params)
        super(InterestUncertainSearch, self).__init__(env, bb_model, obj, params)