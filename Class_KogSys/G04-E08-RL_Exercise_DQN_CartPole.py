"""
Learning to Control the CartPole environment(https://gym.openai.com/envs/CartPole-v0/)
    of OpenAI Gym(https://gym.openai.com/) with a DQN agent.

Recommended tools to install: Anaconda & PyCharm

needs installation of following libraries in the Conda env:
    OpenAI Gym: pip install gym (automatically installs NumPy)
    TensorFlow with Keras: pip install tensorflow

Adopted from [Jon Krohn's code](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/cartpole_dqn.ipynb).

For checking how to implement other RL algorithms refer to:
    Stable Baselines: intensive library providing ready-to-test RL algorithms
    (https://github.com/hill-a/stable-baselines)
"""

import numpy as np
import random  # for choosing a random action in an epsilon greedy policy
from collections import deque  # for creating the experience replay

from tensorflow.keras.models import Sequential  # forward architecture for approximating Q function
from tensorflow.keras.layers import Dense  # only dense layers since state space is with low dimensionality
from tensorflow.keras.optimizers import Nadam  # optimizer for training the network

import os  # for creating directories
import gym  # for creating the CartPole environment

import matplotlib.pyplot as plt


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.experience = deque(maxlen=2000)  # stores the recorded (s,a,r,s') experiences -- elements are added at
        # the end of the list and popped out from the beginning of the list when maxlen reaches
        self.batch_size = 32  # length of each sampled minibatches from the recorded experiences

        self.gamma = 0.95  # discount factor in the expected return

        self.epsilon = 1.0  # exploration rate in the epsilon greedy policy - max exploration when episode begins
        self.epsilon_decay = 0.995  # decay rate for epsilon after each episode termination
        self.epsilon_min = 0.01  # minimum exploration rate for the final epsilon greedy policy

        self.training_rate = 0.001  # training rate of the NN used by the Adam optimizer
        self.model = None

    def policy(self, state):  # agent's behavior policy to choose next action based on current environment state
        raise NotImplementedError()

    def record_experience(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))  # used for minibatch training by
        # experience replay

    def _build_model(self):  # defining the architecture of the NN model
        raise NotImplementedError()

    def train_model(self):  # training NN with sampled minibatches from the recorded (s,a,r,s') experiences
        raise NotImplementedError()

    def save_model(self, name):  # save parameters(weights) of the trained model
        self.model.save_weights(name)

    def load_model(self, name):  # load parameters(weights) of the trained model
        self.model.load_weights(name)


class DQNAgent(Agent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = self._build_model()

    def policy(self, state):
        # exploratory actions by an epsilon greedy policy

        if np.random.rand() <= self.epsilon:  # perform random exploratory action
            return random.randrange(self.action_size)

        # otherwise, exploit the learned target policy to select action
        action_values = self.model.predict(state)  # run NN once in forward direction to approximate Q(s,a)
        return np.argmax(action_values[0])  # choose action a that maximizes Q(s,a)

    def _build_model(self):
        # approximating action-value function by a feed-forward NN with two hidden layers and a linear output layer:
        model = Sequential()

        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Nadam(lr=self.training_rate))
        return model

    def train_model(self):
        batch = random.sample(self.experience, self.batch_size)  # sample a minibatch from the recorded experiences

        for state, action, reward, next_state, done in batch:  # update the model for each sampled experience

            td_target = reward  # if done is true, recorded experience is for the terminating transition of the episode

            if not done:  # otherwise, TD target is approximation of the expected return (greedy policy of Q-Learning)
                td_target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            action_values = self.model.predict(state)
            action_values[0][action] = td_target  # applies the Q-Learning update rule
            self.model.fit(state, action_values, epochs=1, verbose=0)  # updates model to reduce TD error

        if self.epsilon > self.epsilon_min:  # for reducing exploration rate every time model becomes updated
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    output_dir = 'trained_model/'  # directory for saving parameters of the trained model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env = gym.make('CartPole-v0')  # creates an instance object of the CartPole environment

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    num_episodes = 400  # number of episodes to run the CartPole environment from random initial state to termination

    agent = DQNAgent(num_states, num_actions)

    episode_lengths = list()

    for episode in range(num_episodes):

        state_t = env.reset()  # initializes the environment to a random state
        state_t = np.reshape(state_t, [1, num_states])

        terminal = False  # becomes true when episode reaches a terminal state
        time_step = 0  # current time step of the running episode

        while not terminal:
            time_step += 1

            env.render()

            action_t = agent.policy(state_t)  # select action to take based on the agent's behavior policy

            state_tp1, reward_t, terminal, _ = env.step(action_t)  # performing one step simulation of the env

            reward_t = reward_t if not terminal else -10  # actual reward from the CartPole env is only zero when
            # episode fails, it does not punishes too much in the expected return

            state_tp1 = np.reshape(state_tp1, [1, num_states])

            agent.record_experience(state_t, action_t, reward_t, state_tp1, terminal)

            state_t = state_tp1  # updating state to the next one for the upcoming iteration

            if terminal:  # if episode ends:
                episode_lengths.append(time_step)
                print("Episode {:03d}/{} length till termination: {}".format(episode + 1, num_episodes, time_step))

        if len(agent.experience) > agent.batch_size:
            agent.train_model()

        if episode % 30 == 0:
            agent.save_model(output_dir + "weights_{:03d}.hdf5".format(episode))
            # saved agents can be loaded with agent.load_model("./path/filename.hdf5")

    plt.scatter(list(range(num_episodes)), episode_lengths)
    plt.title("Agent's performance over time")
    plt.xlabel('x')
    plt.ylabel('Episode length')
    plt.show()
