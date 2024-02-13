from src.optimization.algs.tree.pareto_mcts import ParetoMCTS
from src.optimization.search.tree.tree_search import TreeSearch


class ParetoMCTSSearch(TreeSearch):

    def __init__(self, env, bb_model, obj, params):
        alg = ParetoMCTS(env, bb_model, obj, params)
        super(ParetoMCTSSearch, self).__init__(env, bb_model, obj, params, alg)


