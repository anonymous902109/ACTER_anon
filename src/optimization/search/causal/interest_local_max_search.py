from src.optimization.algs.causal.interest_local_max_alg import InterestingnessAlgLocalMax
from src.optimization.search.abstract_search import AbstractSearch


class InterestLocalMaxSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgLocalMax(env, bb_model, obj, params)
        super(InterestLocalMaxSearch, self).__init__(env, bb_model, obj, params)