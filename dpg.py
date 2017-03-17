import theano
import numpy as np
from util import  norm_weight, _p, init_uniform, Linear
import theano.tensor as T
import logging
from collections import OrderedDict, namedtuple

seed = 1234
rng = T.shared_randomstreams.RandomStreams(seed)
def param_init_fflayer(options, params, prefix='ff',nin=None, nout=None, ortho=True, flag=False):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
    return params

layers = {'ff': ('param_init_fflayer', 'fflayer')}
def get_layer(name):
         fns = layers[name]
         return (eval(fns[0]), eval(fns[1]))
floatX = theano.config.floatX


def fflayer(tparams, state_below, options, prefix='rconv',
             activ='lambda x: tensor.tanh(x)', **kwargs):
     return T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

class VariableStore(object):
     def __init__(self, prefix="vs", default_initializer=init_uniform()):
         self.prefix = prefix
         self.default_initializer = default_initializer
         self.vars = {}

     @classmethod
     def snapshot(cls, other, name=None):
         name = name or "%s_snapshot" % other.prefix
         vs = cls(name)
         for param_name, param_var in other.vars.iteritems():
             vs.vars[param_name] = theano.shared(param_var.get_value(),
                                                 borrow=False)
         return vs

     def add_param(self, name, shape, initializer=None):
         if initializer is None:
             initializer = self.default_initializer

         if name not in self.vars:
             full_name = "%s/%s" % (self.prefix, name)
             logging.debug("Created variable %s", full_name)
             self.vars[name] = theano.shared(initializer(shape),
                                             name=full_name)

         return self.vars[name]


def init_tparams(params):
     tparams = OrderedDict()
     for kk, pp in params.iteritems():
         tparams[kk] = theano.shared(params[kk], name=kk)
     return tparams

def SGD(cost, params, lr=0.01):
    grads = T.grad(cost, params)
    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        new_values[param] = param - lr * grad
        return new_values

EXPLORE_RANGE = 0.5
Critic = namedtuple("Critic", ["pred", "targets", "cost", "updates"])

def LinearLayer(inp, inp_dim, outp_dim, vs, name="linear", bias=True):
    return Linear(inp, inp_dim, outp_dim, vs, name, bias)

class Model():
    def __init__(self, state_dim, action_dim, explore_range=0.5, name = "dpg", _parent = None, track =True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_range = explore_range
        self.parent = _parent
        self.name = name
        if _parent is None:
            self._vs_actor = VariableStore("%s/vs_a" % name)
            self._vs_critic = VariableStore("%s/vs_c" % name)
            self._vs_prefix = self.name
            self._make_vars()
        else:
             self._vs_actor = VariableStore.snapshot(_parent._vs_actor)
             self._vs_critic = VariableStore.snapshot(_parent._vs_critic)
             self._vs_prefix = _parent.name
             self._pull_vars(_parent)

        self.graph()
        self._updates()
        self._functions()

        if track:
             self.track = Model(state_dim, action_dim, explore_range, track=False, _parent=self, name="%s_track" % name)


    def _make_vars(self):
         self.X = T.matrix("X")
         self.actions = T.matrix("actions")
         self.q_targets = T.vector("q_targets")
         self.lr_actor = T.scalar("lr_actor")
         self.lr_critic = T.scalar("lr_critic")

    def _pull_vars(self, parent):
         self.X = parent.X
         self.actions = parent.actions
         self.q_targets = parent.q_targets
         self.lr_actor = parent.lr_actor
         self.lr_critic = parent.lr_critic
         self.tau = T.scalar("tau")


    def graph(self):
         self.a_pred = LinearLayer(self.X, self.state_dim, self.action_dim, self._vs_actor, name="%s/a" % self._vs_prefix)
         self.a_explore = self.a_pred + rng.normal(self.a_pred.shape, 0, self.explore_range, ndim=2)
         self.critic_given = self.CriticNetwork(self.actions, self.q_targets)
         self.critic_det = self.CriticNetwork(self.a_pred, self.q_targets)
         self.critic_exp = self.CriticNetwork(self.a_explore, self.q_targets)

    def CriticNetwork(self, actions, targets):
         hidden_dim = 200 # DEV
         q_hid = LinearLayer(T.concatenate([self.X, actions], axis=1), self.state_dim + self.action_dim, hidden_dim, self._vs_critic, "%s/q/hid" % self._vs_prefix)
         q_pred = LinearLayer(q_hid, hidden_dim, 1, self._vs_critic, "%s/q/pred" % self._vs_prefix)
         q_pred = q_pred.reshape((-1,))
         q_cost = ((targets - q_pred) ** 2).mean()
         q_updates = SGD(q_cost, self._vs_critic.vars.values(), self.lr_critic)
         return Critic(q_pred, targets, q_cost, q_updates)

    def _updates(self,):
        self.updates = OrderedDict(self.critic_exp.updates)
        self.updates.update(SGD(-self.critic_exp.pred.mean(), self._vs_actor.vars.values(), self.lr_actor))
        if self.parent is not None:
            self.target_updates = OrderedDict()
            for vs, parent_vs in [(self._vs_actor, self.parent._vs_actor), (self._vs_critic, self.parent._vs_critic)]:
                for param_name, param_var in vs.vars.iteritems():
                    self.target_updates[param_var] = (self.tau * vs.vars[param_name] + (1 - self.tau) * parent_vs.vars[param_name])

    def _functions(self):
         self.f_action_on = theano.function([self.X], self.a_pred, allow_input_downcast=True)
         self.f_action_off = theano.function([self.X], self.a_explore, allow_input_downcast=True)
         self.f_q = theano.function([self.X, self.actions], self.critic_given.pred, allow_input_downcast=True)
         self.f_update = theano.function([self.X, self.q_targets, self.lr_actor, self.lr_critic], (self.critic_exp.cost, self.critic_exp.pred), updates=self.updates, allow_input_downcast=True)

         if self.parent is not None:
             self.f_track_update = theano.function([self.tau], updates=self.target_updates, allow_input_downcast=True)


def preprocess_state(state, clip_range=20):
    state = np.clip(state, -clip_range, clip_range)
    return state

def preprocess_action(action, clip_range=20):
    action = np.clip(action, -clip_range, clip_range)
    return action


def update_buffers(buffers, states, actions, rewards, states_next):
     R_states, R_actions, R_rewards, R_states_next = buffers

     R_states = np.append(R_states, states, axis=0)
     R_states_next = np.append(R_states_next, states_next, axis=0)
     R_actions = np.append(R_actions, actions)
     R_rewards = np.append(R_rewards, rewards)

     buffers = (R_states, R_actions, R_rewards, R_states_next)
     return buffers


def run_episode(s, env, buffers, max_len=100,  t = 0, on_policy=True):
     policy = dpg.f_action_on if on_policy else dpg.f_action_off
     def policy_fn(state):
         state = preprocess_state(state).reshape((-1, env.observation_space.shape[0]))
         return policy(state)

     states, actions, rewards, states_next = [], [], [], []
     for j in range(max_len):
        a = policy_fn(np.reshape(s, (1, 3)) )
        s2, r, terminal, info = env.step(a[0])
        states.append(preprocess_state(s))
        actions.append(preprocess_action(a))
        states_next.append(preprocess_state(s2))
        rewards.append(r)

        if terminal:
            break;

     if buffers is not None:
         buffers = update_buffers(buffers, states, actions, rewards, states_next)
     return buffers, (states, actions, rewards, states_next)

def train_batch(dpg, buffers, batch_size, gamma=0.9, tau=0.5, lr_actor=0.01, lr_critic=0.001):
     R_states, R_actions, R_rewards, R_states_next = buffers
     if len(R_states) - 1 < batch_size:
         return 0.0
     idxs = np.random.choice(len(R_states) - 1, size=batch_size, replace=False)
     b_states, _, b_rewards, b_states_next = R_states[idxs], R_actions[idxs], R_rewards[idxs], R_states_next[idxs]
     next_actions = dpg.track.f_action_on(b_states_next)
     b_targets = b_rewards + gamma * dpg.track.f_q(b_states_next, next_actions).reshape((-1,))
     cost_t, _ = dpg.f_update(b_states, b_targets, lr_actor, lr_critic)
     dpg.track.f_track_update(tau)
     return cost_t

import gym
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
STATE_DIM = state_dim
ACTION_DIM = action_dim

R_states = np.empty((0, STATE_DIM), dtype=theano.config.floatX)
R_actions = np.empty((0,), dtype=np.int32)
R_rewards = np.empty((0,), dtype=np.int32)
R_states_next = np.empty_like(R_states)
buffers = (R_states, R_actions, R_rewards, R_states_next)

avg_rewards = []
max_rewards = []
q_costs = []

BATCH_SIZE = 64
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.001
mdp = env

dpg = Model(STATE_DIM, ACTION_DIM, EXPLORE_RANGE,  name = "dpg", _parent = None, track =True)

for t in xrange(50000):
     s = env.reset()
     buffers, _ = run_episode(s, mdp, buffers, 1000, t, on_policy=False)
     cost_t = train_batch(dpg, buffers, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC)
     #s = env.reset()
     _, (states, actions, rewards, _) = run_episode(s, mdp, None, 1000, t, on_policy=True)
     rewards = np.array(rewards)
     avg_reward, max_reward = rewards.mean(), rewards.max()
     avg_rewards.append(avg_reward)
     max_rewards.append(max_reward)
     q_costs.append(cost_t)
     print "%i\t% 4f\t%10f\t\t%f\t%f" % (t, max_reward, cost_t, np.max(states), np.max(actions))
     if not np.isfinite(cost_t):
         print "STOP: Non-finite cost"
         break
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#plt.figure(1)
plt.plot(avg_rewards, "g")
plt.xlabel("Iteration")
plt.ylabel("Average reward achieved")
plt.savefig('average_rewards_same_state_no_exp.png')

#plt.figure(2)
plt.plot(max_rewards, "b")
plt.xlabel("Iteration")
plt.ylabel("Max reward achieved")
plt.savefig('max_rewards_same_state_noexp.png')

#plt.figure(3)
plt.plot(q_costs, "r")
plt.xlabel("Iteration")
plt.ylabel("Q cost")
plt.savefig('q_costs_2_same_state_noexp.png')
