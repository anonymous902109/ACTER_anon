from src.optimization.algs.causal.highlights_alg import HighlightsAlg
from src.optimization.search.abstract_search import AbstractSearch


class HIGHLIGHTSSearch(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        self.alg = HighlightsAlg(env, bb_model, obj, params)
        super(HIGHLIGHTSSearch, self).__init__(env, bb_model, obj, params)