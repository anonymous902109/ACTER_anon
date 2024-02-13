from src.optimization.algs.causal.interest_local_min_alg import InterestingnessAlgLocalMin
from src.optimization.search.abstract_search import AbstractSearch


class InterestLocalMinSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgLocalMin(env, bb_model, obj, params)
        super(InterestLocalMinSearch, self).__init__(env, bb_model, obj, params)