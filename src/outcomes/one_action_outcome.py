from src.outcomes.abstract_outcome import AbstractOutcome


class OneActionOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(OneActionOutcome, self).__init__( bb_model, target_action, true_action)

        self.name = 'why not {}'.format(self.target_action)  # TODO: insert human-readable

    def cf_outcome(self, env, state):
        return self.target_action == self.bb_model.predict(state)  # counterfactual where one specific action is required

    def explain_outcome(self, env, state=None):
        if self.bb_model.predict(state) != self.target_action:  # TODO: reformat true action to something more meaningful
            return True

        return False