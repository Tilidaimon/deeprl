import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards
        # add agents
        num_agents = 3
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
        # add landmarks
        num_landmarks = 3
        angles = np.pi*np.random.rand()
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal = None
            #agent.goal_b = None

        # want other agent to go to the goal landmark
        for agent in world.agents:
            agent.goal = np.random.choice(world.landmarks)
        #world.agents[0].goal_a = world.agents[1]
        #world.agents[0].goal_b = np.random.choice(world.landmarks)
        #world.agents[1].goal_a = world.agents[0]
        #world.agents[1].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for agent in world.agents:
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        #for i, landmark in enumerate(world.landmarks):
        world.landmarks[0].color = np.array([0.75,0.25,0.25]) 
        world.landmarks[1].color = np.array([0.25,0.75,0.25]) 
        world.landmarks[2].color = np.array([0.25,0.25,0.75]) 
        # special colors for goals
        #world.agents[0].goal_a.color = world.agents[0].goal_b.color                
        #world.agents[1].goal_a.color = world.agents[1].goal_b.color                               
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.dead = False

    def reward(self, agent, world):
        rew = 0;
        if agent.goal is None: #or agent.goal_b is None:
            return 0.0
        if agent.collide:
            for a in world.agents:
                if agent==a: continue
                if self.is_collision(a, agent):
                    rew -= 10
        if self.is_collision(agent.goal, agent):
            if not agent.goal.dead:
                rew += 10
                for landmark in world.landmarks:
                    if landmark == agent.goal:
                        landmark.dead = True


        rew -= np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
        return rew

    def observation(self, agent, world):
        # goal color
        #goal_color = [np.zeros(world.dim_color), np.zeros(world.dim_color)]
        #if agent.goal_b is not None:
        #    goal_color[1] = agent.goal_b.color 

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        #entity_color = []
        #for entity in world.landmarks:
        #    entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        return np.concatenate([agent.state.p_vel] + entity_pos + comm)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
            
    def is_done(self, agent):
        return agent.dead