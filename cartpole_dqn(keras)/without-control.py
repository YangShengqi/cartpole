import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action



# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print (sess.run(hello))

# a = tf.constant(10)
# b = tf.constant(32)
# print (sess.run(a+b))