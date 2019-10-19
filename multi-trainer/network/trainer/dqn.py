import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import gym
 
# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
# 强化学习的参数
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
# 导入实验环境
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
 
 
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, biflag):
        super(ActorNet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        if biflag:
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biflag = biflag
        #self.device = torch.device("cuda")

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=biflag)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num, output_dim),
            nn.LogSoftmax()
            )
        self.hidden = self.init_hidden(self.batch_size)
        #self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        #self.out = nn.Linear(10, N_ACTIONS)


    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.bi_num, self.batch_size, self.hidden_dim),#.to(self.device),
            torch.zeros(self.num_layers*self.bi_num, self.batch_size, self.hidden_dim))#.to(self.device))
        
    def forward(self, x):
        #self.hidden = self.init_hidden(self.batch_size)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.out(lstm_out)
        return out

class DQN(object):
    def __init__(self, obs_space, hidden_dim, action_space, num_layers, batch_size, biflag):
        self.obs_space = obs_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.eval_net = Net(self.obs_space, self.hidden_dim, self.action_space, num_layers, self.batch_size, biflag)
        self.target_net = Net(self.obs_space, self.hidden_dim, self.action_space, num_layers, self.batch_size, biflag)
        # 记录学习到多少步
        self.learn_step_counter = 0  # for target update
        #self.memory_counter = 0  # for storing memory
        # 初始化memory
        self.replay_buffer = ReplayBuffer(1e6)
        self.batch_size = args.batch_size
        self.max_episode_len = args.max_episode_len
        self.max_replay_buffer_len = self.batch_size * self.max_episode_len
        self.replay_sample_index = None

        #self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.e_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.t_optimizer = torch.optim.Adam(self.target_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
 
    def choose_action(self, obs, epsilon):
        obs = Variable(torch.unsqueeze(torch.FloatTensor(obs), 0))
        if np.random.uniform() > epsilon:
            action_value = self.eval_net.forward(obs)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else: # random
            action = np.random.randint(0, self.action_space)
        return action
 

    def store_transaction(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None
 
    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
 
        #sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        """
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES: N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES: ]))
        """
        obs_n = []
        obs_next_n = []
        act_n = []
        rew_n = []
        index = self.replay_sample_index
        obs_n, act_n, rew_n, obs_next_n, done_n = self.replay_buffer.sample_index(index)

        obs_n = torch.FloatTensor(obs_n)
        act_n = torch.FloatTensor(act_n)
        rew_n = torch.FloatTensor(rew_n)
        obs_next_n = torch.FloatTensor(obs_next_n)

        q_eval = self.eval_net(obs_n).gather(1, act_n)
        q_next = self.target_net(obs_next_n).detach()
        q_target = rew_n + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
 


