class Trajectory:

    def __init__(self, outcome, id=0):
        self.id = id
        self.states = []
        self.actions = []
        self.env_states = []
        self.outcome = []
        self.end_state = []
        self.start_state = []
        self.outcome = outcome

    def to_string(self):
        pass

    def append(self, state, action, env_unwrapped):
        self.states.append(state)
        self.actions.append(action)
        self.env_states.append(env_unwrapped)

    def set_end_state(self, end_state):
        self.end_state = end_state