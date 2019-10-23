import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time

#from trainer.common.distributions import make_pdtype
#from trainer import AgentTrainer
from birnn.trainer.replay_buffer import ReplayBuffer


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class ActorNet(nn.Module):   # ae(s)=a
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biflag):
        super(ActorNet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        if biflag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biflag = biflag
        self.device = torch.device("cuda")

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biflag)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num, output_dim),
            nn.LogSoftmax()
            )

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers*self.bi_num, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers*self.bi_num, batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        hidden = self.init_hidden(self.batch_size)
        y = self.lstm(x)
        out = self.out(y)
        return out

class CriticNet(nn.Module):   # ae(s)=a
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biflag):
        super(CriticNet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        if biflag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biflag = biflag
        self.device = torch.device("cuda")

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biflag)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num, output_dim),
            nn.LogSoftmax()
            )

    def init_hidden(self.batch_size):
        return (torch.zeros(self.num_layers*self.bi_num, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers*self.bi_num, batch_size, self.hidden_dim).to(self.device))

    def forward(self,x):
        y = self.lstm(x)
        out = self.out(y)
        return out


class DDPG(object):
    def __init__(self, obs_space, act_space, args):
        self.a_input_dim = obs_space
        self.a_output_dim = act_space
        self.c_input_dim = obs_space
        self.c_output_dim = 1

        self.replay_buffer = ReplayBuffer(1e6)
        self.batch_size = args.batch_size
        self.max_episode_len = args.max_episode_len
        self.max_replay_buffer_len = self.batch_size * self.max_episode_len
        self.replay_sample_index = None

        #self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ActorNet(self.a_input_dim, self.a_output_dim)
        self.Actor_target = ActorNet(self.a_input_dim, self.a_output_dim)
        self.Critic_eval = CriticNet(self.c_input_dim, self.c_output_dim)
        self.Critic_target = CriticNet(self.c_input_dim, slef.c_output_dim)

        self.c_opt = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.a_opt = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach() # ae（s）

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def learn(self, agents, t):
        """
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')
        """
        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return


        self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        rew_n = []
        index = self.replay_sample_index
        for i in range(self.obs_space):
            obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            rew_n.append(rew)
            act_n.append(act)
        #obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        obs_n = torch.FloatTensor(obs_n)
        act_n = torch.FloatTensor(act_n)
        rew_n = torch.FloatTensor(rew_n)
        obs_next_n = torch.FloatTensor(obs_next_n)

        for i in len(agents):
            action, hidden = self.Actor_eval(obs_n, hidden)
        q = self.Critic_eval(obs_n)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) 
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
output_dim = env.observation_space.shape[0]
input_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(input_dim, output_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)