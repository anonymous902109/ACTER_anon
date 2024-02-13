from src.outcomes.abstract_outcome import AbstractOutcome


class OneActionOutcome(AbstractOutcome):

    def __init__(self):
        super(OneActionOutcome, self).__init__()
        self.target_outcome = None
        self.failure_outcome = None

    def cf_outcome(self, env, bb_model, state, cf_action):
        return cf_action == bb_model.predict(state)  # counterfactual where one specific action is required