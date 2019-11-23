import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import numpy as np
import random
import gym


LR = 0.01
GAMMA = 0.95
EPSILON_INIT = 1
EPSILON_DECAY = 0.9
EPSILON_FIN = 0.01
STATE_SIZE = 4
ACTION_SIZE = 2
BATCH_SIZE = 32


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class DQNAgent:

    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.net = Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        self.epsilon = EPSILON_INIT

    def act(self, state):
        state = torch.Tensor(state).view(-1, 4)
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(ACTION_SIZE)
        else:
            Q_tensor = self.net(state)
            Q_array = Q_tensor.detach().numpy()
            action = np.argmax(Q_array[0])
            #action = torch.max(action_value, 1)[1].numpy()[0, 0]
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        minibatch = random.sample(self.memory, BATCH_SIZE)

        Qt_batch = np.zeros([BATCH_SIZE, 2])
        state_batch = np.zeros([BATCH_SIZE, 4])
        i = 0

        for state, action, reward, state_n, done in minibatch:
            state_batch[i] = state
            state_tensor = torch.Tensor(state).view(-1, 4)
            Q_tensor = self.net(state_tensor)
            Q_array = Q_tensor.detach().numpy()
            Qt_batch[i] = Q_array[0]
            if done:
                Qt_batch[i][action] = reward
            else:
                state_n_tensor = torch.Tensor(state_n).view(-1, 4)
                Q_n_tensor = self.net(state_n_tensor)
                Q_n_array = Q_n_tensor.detach().numpy()
                Qt_batch[i][action] = reward + GAMMA * np.amax(Q_n_array[0])
            i += 1

        state_batch_tensor = torch.Tensor(state_batch).view(-1, 4)
        Qt_batch_tensor = torch.Tensor(Qt_batch).view(-1, 2)
        self.optimizer.zero_grad()
        Q_batch_tensor = self.net(state_batch_tensor)

        loss = self.loss(Q_batch_tensor, Qt_batch_tensor)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_FIN:
            self.epsilon *= EPSILON_DECAY


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state = env.reset()
    agent = DQNAgent()
    score = 0
    episode = 0
    agent.net.load_state_dict(torch.load('train_weights.pth'))
    while True:
        #env.render()
        action = agent.act(state)
        state_n, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, state_n, done)
        state = state_n
        score += reward
        if done:
            episode += 1
            print("Episode: %d | Score: %d" % (episode, score))
            if len(agent.memory) > BATCH_SIZE:
                agent.train()
            score = 0
            state = env.reset()
            if episode > 0 and episode % 50 == 0:
                torch.save(agent.net.state_dict(), 'train_weights.pth')
