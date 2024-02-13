import copy

from src.optimization.objs.back_obj import BackObj


class SemifactObj(BackObj):

    def __init__(self, env, bb_model, params):
        super(SemifactObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'proximity', 'sparsity']
        self.constraints = ['validity']


    def get_objectives(self, fact, cf, actions, target_action):
        proximity = self.action_proximity(fact, actions)
        sparsity = self.sparsity(fact, actions)
        stochasticity = self.stoch_validity(fact, actions)

        return {'uncertainty': stochasticity,
                'proximity': proximity,
                'sparsity': sparsity}

    def get_constraints(self, fact, cf, actions, target_action):
        validity = self.validity(fact, actions)

        return {'validity': validity}

    def validity(self, fact, actions):
        self.env.reset()
        self.env.set_stochastic_state(copy.deepcopy(fact.env_states[0]))
        for a in actions:
            _, _, done, trunc, _ = self.env.step(a)
            if done or trunc or self.env.check_failure():
                break

        return not self.env.check_failure()

    def stoch_validity(self, fact, actions):
        n_sim = 10
        cnt = 0
        for i in range(n_sim):
            self.env.reset()
            self.env.set_nonstoch_state(copy.deepcopy(fact.env_states[0]))
            for a in actions:
                _, _, done, trunc, _ = self.env.step(a)
                if done or trunc:
                    break

            if self.env.check_failure():
                cnt += 1

        return 1 - (cnt * 1.0)/n_sim

    def action_proximity(self, fact, actions):
        avg_distance = super().action_proximity(fact, actions)

        return 1 - avg_distance  # Goal to maximize distance


