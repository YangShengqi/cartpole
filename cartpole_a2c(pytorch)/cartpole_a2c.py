import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random


STATE_SIZE = 4
ACTION_SIZE = 2
BATCH_SIZE = 32
GAMMA = 0.99


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, ACTION_SIZE)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ACAgent:

    def __init__(self):
        self.critic_net = CriticNet()
        self.actor_net = ActorNet()
        self.critic_net_optim = optim.Adam(self.critic_net.parameters(), lr=0.01)
        self.actor_net_optim = optim.Adam(self.actor_net.parameters(), lr=0.01)
        self.critic_net_loss = 0
        self.actor_net_loss = 0
        self.memory = deque(maxlen=2000)
        self.states = []
        self.onehot_actions = []
        self.rewards = []

    def act(self, state):
        log_policy = self.actor_net(torch.Tensor(state))
        policy = torch.exp(log_policy)
        action = np.random.choice(ACTION_SIZE, 1, p=policy.data.numpy())[0]
        return action

    def sample(self, state, action, reward, state_n, done):
        self.memory.append((state, action, reward, state_n, done))
        onehot_action = np.zeros(ACTION_SIZE)
        onehot_action[action] = 1
        self.onehot_actions.append(onehot_action)
        self.states.append(state)
        self.rewards.append(reward)

    def cal_q(self):
        q = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = self.rewards[t] + GAMMA * running_add
            q[t] = running_add
        return q

    def train(self):
        states_tsr = torch.Tensor(self.states).view(-1, STATE_SIZE)
        onehot_actions_tsr = torch.Tensor(self.onehot_actions).view(-1, ACTION_SIZE)

        # train actor network
        self.actor_net_optim.zero_grad()
        log_policys = self.actor_net(states_tsr)
        vs = torch.squeeze(self.critic_net(states_tsr).detach())
        qs = torch.Tensor(self.cal_q())
        advantages = qs - vs
        self.actor_net_loss = -torch.mean(torch.sum(log_policys*onehot_actions_tsr, 1) * advantages)
        self.actor_net_loss.backward()
        self.actor_net_optim.step()

        # train critic network
        minibatch = random.sample(self.memory, BATCH_SIZE)
        s_batch = np.zeros([BATCH_SIZE, STATE_SIZE])
        vt_batch = torch.zeros(BATCH_SIZE)
        i = 0
        for state, action, reward, state_n, done in minibatch:
            s_batch[i] = state
            if done:
                vt_batch[i] = 0
            else:
                v_n = torch.squeeze(self.critic_net(torch.Tensor(state_n).view(-1, 4)).detach())
                vt_batch[i] = reward + GAMMA * v_n
            i += 1
        s_batch = torch.Tensor(s_batch).view(-1, 4)
        v_batch = torch.squeeze(self.critic_net(s_batch))

        self.critic_net_optim.zero_grad()
        criterion = nn.MSELoss()
        self.critic_net_loss = criterion(v_batch, vt_batch)
        self.critic_net_loss.backward()
        self.critic_net_optim.step()
        self.states, self.onehot_actions, self.rewards = [], [], []


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    state = env.reset()
    score = 0
    episode = 0
    agent = ACAgent()
    #agent.actor_net.load_state_dict(torch.load('actor_weights'))
    #agent.critic_net.load_state_dict(torch.load('critic_weights'))
    while True:
        action = agent.act(state)
        state_n, reward, done, _ = env.step(action)
        agent.sample(state, action, reward, state_n, done)
        state = state_n
        score += reward

        if done:
            episode += 1
            if len(agent.memory) > BATCH_SIZE:
                agent.train()
            state = env.reset()
            if episode > 0 and episode % 50 == 0:
                print('Episode: %d | Score: %d' % (episode, score/50.0))
                torch.save(agent.actor_net.state_dict(), 'actor_weights')
                torch.save(agent.critic_net.state_dict(), 'critic_weights')
                score = 0

