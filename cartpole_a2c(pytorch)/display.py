from cartpole_a2c import ACAgent
import gym
import torch


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = ACAgent()
    agent.actor_net.load_state_dict(torch.load('actor_weights'))
    agent.critic_net.load_state_dict(torch.load('critic_weights'))
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

