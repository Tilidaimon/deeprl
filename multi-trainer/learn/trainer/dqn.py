import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import gym
from learn.trainer.replay_buffer import ReplayBuffer
 
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
        super(Net,self).__init__()
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

        #print(self.bi_num, self.hidden_dim)
        #print(self.hidden_dim*self.bi_num)
        #print(self.output_dim)
        self.out = nn.Linear(self.hidden_dim*self.bi_num, self.output_dim)

        #self.hidden = self.init_hidden()
        self.lstm.all_weights[0][0].data.normal_(0, 0.1)  # 初始化
        self.lstm.all_weights[0][1].data.normal_(0, 0.1)  # 初始化
        self.lstm.all_weights[1][0].data.normal_(0, 0.1)  # 初始化
        self.lstm.all_weights[1][1].data.normal_(0, 0.1)  # 初始化
        #self.out = nn.Linear(10, N_ACTIONS)


    def init_hidden(self):
        return (torch.zeros(self.num_layers*self.bi_num, self.batch_size, self.hidden_dim, requires_grad=True),#.to(self.device),
            torch.zeros(self.num_layers*self.bi_num, self.batch_size, self.hidden_dim, requires_grad=True))#.to(self.device))
        
    def forward(self, x):
        #self.hidden = self.init_hidden(self.batch_size)
        #hidden = self.init_hidden()

        #h = [x.permute(1, 0, 2).contiguous() for x in hidden]
        lstm_out, _ = self.lstm(x)

        #hidden = [x.permute(1, 0, 2).contiguous() for x in h]
        #print('hidden:',len(hidden[0].size()))
        
        out = self.out(lstm_out)
        #print('out',out.shape)
        return out

class DQN(object):
    def __init__(self, obs_space, action_space, hidden_dim, num_layers, batch_size, biflag, args):
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
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        #self.t_optimizer = torch.optim.Adam(self.target_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
 
    def choose_action(self, obs, num_agents, epsilon):
        #print('obs',obs)
        #print(obs.size())
        obs = Variable(torch.unsqueeze(torch.FloatTensor(obs), 0))
        print('obs',obs.size())

        if np.random.uniform() > epsilon:
            action_value = self.eval_net.forward(obs)
            print('select max action')
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            action = action.reshape(num_agents,1)
            #if len(action)>1:
            #print(action_value, action)
        else: # random
            print('random select')
            action = np.random.randint(self.action_space, size=(num_agents,1))
        print(action)

        return action
 

    def store_transaction(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        #print(obs,'\n', act,'\n', rew,'\n', new_obs,'\n', done)
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
        
        obs_n = Variable(torch.FloatTensor(obs_n))
        print(act_n.shape, act_n)
        act_n = torch.from_numpy(act_n)
        rew_n = Variable(torch.FloatTensor(rew_n))
        obs_next_n = Variable(torch.FloatTensor(obs_next_n))

        #print('obs_n:',obs_n.size())
        #print('obs_next:',obs_next_n.size())

        #print('act_n:',act_n,act_n.size())
        
        q_eval = self.eval_net(obs_n.view(len(index),1,-1))
        q_eval = q_eval.view(len(index),-1)
        #print('q_eval',q_eval,q_eval.size())
        q_eval = q_eval.gather(1, act_n)
        q_next = self.target_net(obs_next_n.view(len(index),1,-1)).detach()
        q_next = q_next.view(len(index),-1)
        #print('q_next',q_next,q_next.size())
        #print('rew_n',rew_n,rew_n.size())
        #print(q_next.max(1))
        #print(q_next.max(1)[0].size())
        q_target = rew_n + GAMMA * q_next.max(1)[0]
        #print(q_target,q_target.size())
        loss = self.loss_func(q_eval.float(), q_target.float())
        #print('loss',loss)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
 
 


