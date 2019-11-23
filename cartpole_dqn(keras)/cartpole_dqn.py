import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential, load_model


class DQNAgent:
    def __init__(self, nb_state, nb_action):
        self.nb_state = nb_state
        self.nb_action = nb_action
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.model = self.build()

    def build(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.nb_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nb_action)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch):
        minibatch = random.sample(self.memory, batch)
        for state, action, reward, next_state, done in minibatch:
            qvalue_tar = reward
            if not done:
                qvalue_tar = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            qvalue_pre = self.model.predict(state)
            qvalue_pre[0][action] = qvalue_tar
            self.model.fit(state, qvalue_pre, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    episodes = 10000
    batch = 32
    env = gym.make('CartPole-v1')
    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n
    agent = DQNAgent(nb_state, nb_action)
    # agent.load('train.h5')
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, nb_state])
        for score_t in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, nb_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: %d / %d, score: %d" % (e, episodes, score_t))
                break
        if len(agent.memory) > batch:
            agent.replay(batch)
        agent.save('train.h5')


