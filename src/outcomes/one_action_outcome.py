from src.outcomes.abstract_outcome import AbstractOutcome


class OneActionOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(OneActionOutcome, self).__init__( bb_model, target_action, true_action)

        self.name = 'any_{}'.format(self.true_action)

    def cf_outcome(self, env, state):
        # return cf_action == bb_model.predict(state)  # counterfactual where one specific action is required
        pass