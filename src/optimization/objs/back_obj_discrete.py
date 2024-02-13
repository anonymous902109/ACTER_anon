import copy
import math
import torch

from src.approaches.models.enc_dec import EncoderDecoder
from src.optimization.objs.back_obj import BackObj


class BackObjDiscrete(BackObj):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, params):

        super(BackObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'proximity', 'sparsity', 'recency']
        self.constraints = ['validity']

        self.n_sim = params['n_sim']

        self.enc_dec = EncoderDecoder(self.env, self.bb_model, path='../datasets/{}/'.format(params['task_name']), k=params['horizon'])

    def action_proximity(self, fact, actions):
        fact_traj = self.combine(fact.states, fact.actions)
        cf_traj = self.get_trajectory(fact, actions)

        fact_enc = self.enc_dec.encode(fact_traj)
        cf_enc = self.enc_dec.encode(cf_traj)

        distance = math.sqrt(sum((fact_enc - cf_enc)**2))

        return distance

    def combine(self, states, actions):
        ''' Combines a list of states and actions into format (s1, ..., sn, a1, ... , an) '''
        comb = []
        for s in states:
            # all all states first
            comb.extend(list(s.flatten()))

        for a in actions:
            # add all actions
            comb.append(a)

        return torch.tensor(comb)

    def get_trajectory(self, fact, actions):
        self.env.reset()
        self.env.set_stochastic_state(copy.copy(fact.states[0]), copy.deepcopy(fact.env_states[0]))

        t = []
        t.extend(list(fact.states[0].flatten()))
        i = 0
        for a in actions:
            obs, _, done, trunc, _ = self.env.step(a)
            if (i != len(actions) - 1):
                t.extend(list(obs.flatten())) # don't add the last state
            i += 1
            if done or trunc or self.env.check_failure():
                break

        while i < (len(actions) - 1):
            # not all actions have been executed because of validity being broken
            t.extend(list(self.env.reset()[0].flatten()))  # add a random state to fill up space
            i += 1

        for a in actions:
            t.append(a)

        return torch.tensor(t)


