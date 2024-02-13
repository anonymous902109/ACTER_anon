class AbstractOutcome():

    def __init__(self):
        self.target_outcome = None
        self.failure_outcome = None

    def cf_outcome(self, env, bb_model, state, action):
        return True