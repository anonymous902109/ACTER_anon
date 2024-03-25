import copy

from src.optimization.objs.explaining.abstract_obj import AbstractObjective


class CfExplObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, params):

        super(CfExplObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'proximity', 'sparsity']
        self.constraints = ['validity', 'fidelity']

        self.n_sim = params['n_sim']

    def get_constraints(self, fact, cf, actions, target_action):
        validity = self.validity(fact, actions)
        fidelity = self.fidelity(fact, actions, self.bb_model)

        return {'validity': validity,
                'fidelity': fidelity}

    def get_objectives(self, fact, cf, actions, target_action):
        proximity = self.action_proximity(fact, actions)
        sparsity = self.sparsity(fact, actions)
        stochasticity = self.stoch_validity(fact, actions)

        return {'uncertainty': stochasticity,
                'proximity': proximity,
                'sparsity': sparsity}

    def get_first_state(self, fact):
        return copy.copy(fact.states[0]), copy.deepcopy(fact.env_states[0])



