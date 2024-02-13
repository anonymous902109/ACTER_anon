from src.optimization.algs.causal.interest_freq_alg import InterestingnessAlgFreq
from src.optimization.search.abstract_search import AbstractSearch


class InterestFreqSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgFreq(env, bb_model, obj, params)
        super(InterestFreqSearch, self).__init__(env, bb_model, obj, params)