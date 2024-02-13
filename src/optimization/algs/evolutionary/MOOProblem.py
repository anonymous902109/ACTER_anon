from src.optimization.algs.evolutionary.evol_problem import EvolutionaryProblem


class MOOProblem(EvolutionaryProblem):

    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, fact, obj):
        super().__init__(n_var, n_obj, n_ieq_constr, xl, xu, fact, obj)

    def fitness_func(self, solution):
        solution = list(solution)

        if tuple(solution) in self.prev_solutions.keys():
            fitness, constraint = self.prev_solutions[tuple(solution)]

            return fitness, constraint

        output = self.obj.get_objectives(self.fact, None, solution, None)
        fitness = [output[obj_name] for obj_name in self.obj.objectives]

        constraint_dict = self.obj.get_constraints(self.fact, None, solution, None)
        constraints = [int(constraint_dict[c_name]) for c_name in self.obj.constraints]

        self.prev_solutions[tuple(solution)] = (fitness, constraints)

        output.update(constraint_dict)
        self.rew_dict[tuple(solution)] = output

        return fitness, constraints