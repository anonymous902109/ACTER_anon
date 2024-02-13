class Trajectory:

    def __init__(self, id=0):
        self.id = id
        self.states = []
        self.actions = []
        self.env_states = []

    def to_string(self):
        pass

    def append(self, state, action, env_unwrapped):
        self.states.append(state)
        self.actions.append(action)
        self.env_states.append(env_unwrapped)