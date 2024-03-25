import copy

from src.optimization.objs.explaining.abstract_obj import AbstractObjective


class PfDiagObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, params):

        super(PfDiagObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'reachability']
        self.constraints = ['validity']

        self.n_sim = params['n_sim']

    def get_constraints(self, fact, cf, actions, target_action):
        actions = [a for a in actions if a != self.noop]  # remove Noop actions
        validity = self.validity(fact, actions)

        return {'validity': validity}

    def get_objectives(self, fact, cf, actions, target_action):
        actions = [a for a in actions if a != self.noop]  # remove Noop actions

        reachability = self.reachability(actions)
        stochasticity = self.stoch_validity(fact, actions)

        return {'uncertainty': stochasticity,
                'reachability': reachability}

    def get_first_state(self, fact):
        return copy.copy(fact.states[-1]), copy.deepcopy(fact.env_states[-1])



