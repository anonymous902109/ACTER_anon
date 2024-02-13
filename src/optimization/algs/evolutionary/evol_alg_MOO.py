from src.optimization.algs.evolutionary.MOOProblem import MOOProblem
from src.optimization.algs.evolutionary.evol_alg import EvolutionaryAlg


class EvolutionAlgMOO(EvolutionaryAlg):

    def __init__(self, env, bb_model, obj, params):
        super(EvolutionAlgMOO, self).__init__(env, bb_model, obj, params)

    def generate_problem(self, fact):
        n_objectives = len(self.obj.objectives)
        n_constraints = len(self.obj.constraints)
        return MOOProblem(self.horizon, n_objectives, n_constraints, self.xl, self.xu, fact, self.obj)


