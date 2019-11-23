import gym
import numpy as np
from dqn_cartpole import DQNAgent


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n
    agent = DQNAgent(nb_state, nb_action)
    agent.load('train.h5')
    state = env.reset()
    state = np.reshape(state, [1, nb_state])
    for score_t in range(1000):
        env.render()
        act_values = agent.model.predict(state)
        action = np.argmax(act_values[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, nb_state])
        state = next_state
        if done:
            print('score: %d' % (score_t,))
            break
