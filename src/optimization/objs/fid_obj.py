import copy

from src.optimization.objs.abs_obj import AbstractObj


class FidObj(AbstractObj):
    '''
    Set of objectives used by the RACCER algorithm for generating prefactuals
    '''

    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']
        self.constraints = ['validity']
        super(FidObj, self).__init__(env, bb_model, params)

    def get_objectives(self, fact, cf, actions, target_action):
        stochasticity, validity, fidelity = self.calculate_stochastic_rewards(fact, actions, target_action, self.bb_model)

        reachability = self.reachability(actions)

        return {
            'fidelity': fidelity,
            'reachability': reachability,
            'stochastic_validity': validity
        }

    def get_constraints(self, fact, cf, actions, target_action):
        validity = self.validity(fact, actions)

        return {'validity': validity}

    def validity(self, fact, actions):
        self.env.reset()
        self.env.set_stochastic_state(copy.copy(fact.states[0]), copy.deepcopy(fact.env_states[0]))
        for a in actions:
            obs, _, done, trunc, _ = self.env.step(a)
            if done or trunc or self.env.check_failure():
                break
        valid_outcome = fact.outcome.cf_outcome(self.env, obs)

        # IMPORTANT: return 1 if the class hasn't changed -- to be compatible with minimization used by NSGA
        return not valid_outcome