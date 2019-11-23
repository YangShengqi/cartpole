import gym
from cartpole_pg import PGAgent


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    agent.load('train_weights_150.h5')
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
