from src.outcomes.abstract_outcome import AbstractOutcome


class ActionOutcome(AbstractOutcome):

    def __init__(self):
        super(ActionOutcome, self).__init__()
        self.target_outcome = None
        self.failure_outcome = None

    def cf_outcome(self, env, bb_model, state, action):
        return action != bb_model.predict(state) # counterfactual where any alternative action is chosen