import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.states = []
        self.rewards = []
        self.labels = []
        self.model = self.buildmodel()

    def buildmodel(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        prob = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action

    def remember(self, state, action, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.labels.append(y)
        self.states.append(state)
        self.rewards.append(reward)

    def discount_rewards(self, rewards):
        gammas = np.zeros_like(rewards)
        gamma = 1
        for i in range(len(gammas)):
            gammas[i] = gamma
            gamma *= self.gamma
        discounted_rewards = rewards * gammas
        summary = np.sum(discounted_rewards)
        discounted_rewards = np.full(rewards.shape, summary)
        return discounted_rewards

    def train(self):
        states = np.vstack(self.states)
        labels = np.vstack(self.labels)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        labels *= -rewards
        X = states
        Y = labels
        self.model.train_on_batch(X, Y)
        self.states, self.labels, self.rewards = [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state = env.reset()
    score = 0
    episode = 0
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    #agent.load('train_weights_6000')
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward)
        state = next_state
        score += reward

        if done:
            episode += 1
            agent.train()
            print('Episode: %d | Score: %d' % (episode, score))
            score = 0
            state = env.reset()
            if episode > 0 and episode % 50 == 0:
                agent.save('train_weights.h5')
