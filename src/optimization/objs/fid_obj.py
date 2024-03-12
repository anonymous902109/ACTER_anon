from src.optimization.objs.abs_obj import AbstractObj


class FidObj(AbstractObj):
    '''
    Set of objectives used by the RACCER algorithm for generating prefactuals
    '''

    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']
        self.constraints = []
        super(FidObj, self).__init__(env, bb_model, params)

    def get_objectives(self, fact, cf, actions, target_action):
        stochasticity, validity, fidelity = self.calculate_stochastic_rewards(fact, actions, target_action, self.bb_model)

        reachability = self.reachability(actions)

        return {
            'fidelity': fidelity,
            'reachability': reachability,
            'stochastic_validity': validity
        }