from src.optimization.algs.causal.intrest_no_freq_alg import InterestingnessAlgNoFreq
from src.optimization.search.abstract_search import AbstractSearch


class InterestNoFreqSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = InterestingnessAlgNoFreq(env, bb_model, obj, params)
        super(InterestNoFreqSearch, self).__init__(env, bb_model, obj, params)