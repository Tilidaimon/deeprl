import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

#import maddpg.common.tf_util as U
from learn.trainer.dqn import DQN
#import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagentenvs.environment import MultiAgentEnv
    import multiagentenvs.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env



def train(arglist):
    print('Start training')
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)

    input_dim = env.observation_dim
    output_dim = env.action_space
    print('build network')
    dqn = DQN(input_dim=input_dim, output_dim=output_dim, hidden_dim=20, num_layers=3, batch_size=arglist.batch_size, biflag=True)

    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore or arglist.benchmark:
        print('Loading previous state...')
        U.load_state(arglist.load_dir)
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    saver = tf.train.Saver()
    print('reset envs')
    obs = env.reset()
    print('Start iteration...')
    while True:
        env.render()
 
        act = dqn.choose_action(obs)
        # take action
        obs_next, rew,  done, info = env.step(act)
        terminal = (episode_step >= arglist.max_episode_len)

        for r in rew:
            episode_rewards[-1] += r
            #agent_rewards[i][-1] += rew
 
        dqn.store_transaction(obs, act, rew, obs_next, done)
 

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            #agent_info.append([[]])

         # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        dqn.preupdate()
        dqn.learn()
        obs = obs_next

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            U.save_state(arglist.save_dir, saver=saver)
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
