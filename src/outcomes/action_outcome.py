from src.outcomes.abstract_outcome import AbstractOutcome


class ActionOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(ActionOutcome, self).__init__( bb_model, target_action, true_action)

        self.name = 'change_{}_to_{}'.format(self.true_action, self.target_action)

    def cf_outcome(self, env, state):
        # return action != bb_model.predict(state) # counterfactual where any alternative action is chosen
        pass