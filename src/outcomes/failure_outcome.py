from src.outcomes.abstract_outcome import AbstractOutcome


class FailureOutcome(AbstractOutcome):

    def __init__(self):
        super(FailureOutcome, self).__init__()
        self.target_outcome = None
        self.failure_outcome = None

    def cf_outcome(self, env, bb_model, state, cf_action):
        return not env.check_failure()  # counterfactual where one specific action is required

    def explain_outcome(self, env, bb_model=None, state=None, cf_action=None):
        return env.check_failure()  # if failure explain this outcome