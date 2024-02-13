from src.optimization.algs.causal.interest_certain_alg import InterestingnessAlgCertain
from src.optimization.search.abstract_search import AbstractSearch


class InterestCertainSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgCertain(env, bb_model, obj, params)
        super(InterestCertainSearch, self).__init__(env, bb_model, obj, params)