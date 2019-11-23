from test import DQNAgent
import gym
import torch


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    agent.epsilon = 0
    agent.net.load_state_dict(torch.load('train_weights.pth'))
    state = env.reset()
    score = 0
    while True:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += 1
        if done:
            print('score: %d' % (score,))
            break
