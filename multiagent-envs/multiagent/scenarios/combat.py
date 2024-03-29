import numpy as np
from multiagentenvs.core import World, Agent, Landmark
from multiagentenvs.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.state.p_angle = np.random.uniform(-2*np.pi, 2*np.pi)
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = 0.1*np.ones(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_angle = np.random.uniform(-2*np.pi, 2*np.pi)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        min_dist = 1e6
        los = agent.target.state.p_pos - agent.state.p_pos
        dist = np.sqrt(np.sum(np.square(los)))
        v = np.sqrt(np.sum(np.square(agent.state.p_vel)))
        
        agent_angle = np.dot(los, agent.state.p_vel)/(v*dist)
        a = [np.cos(agent.target.state.p_angle),np.sin(agent.target.state.p_angle)]
        target_angle = np.dot(-los, a)/(dist)

        rew -= (target_angle - agent_angle)/dist

        if self.is_collision(agent.target, agent):
            if agent.target.dead == False:
                rew += 10
                agent.target.dead = True
        else:
            if agent.target.dead == True:
                rew += 5

        if agent.collide:
            for a in world.agents:
                if a==agent: continue
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        target_pos = []
        agent_angles = []
        target_angles = []
        for target in world.landmarks:  # world.entities:
            los = target.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(los)))
            v = np.sqrt(np.sum(np.square(agent.state.p_vel)))
            target_pos.append(dist)
            #print('los', los, 'agent', agent.state.p_vel, 'v', v, 'dist', dist)
            try:
                agent_angle = np.dot(los, agent.state.p_vel)/(v*dist)
            except RuntimeWarning:
                print('los', los, 'agent', agent.state.p_vel, 'v', v, 'dist', dist)
            a = [np.cos(target.state.p_angle),np.sin(target.state.p_angle)]
            target_angle = np.dot(-los, a)/(dist)
            target_pos.append(los)
            agent_angles.append(agent_angle)
            target_angles.append(target_angles)
        # communication of all other agents
        comm = []
        other_pos = []
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + agent_angles + target_pos + target_angles)
