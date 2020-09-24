import gym
# import math
import random
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# from collections import deque
# from collections import namedtuple
# from itertools import count
# from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

SCORES = []

CYCLES = 1000

GAMMA = 0.98                # Amount to discount future rewards.
LEARNING_RATE = 0.005       # Speed at which weight changes.

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQN(nn.Module):
    def __init__(self, observation_space, num_actions, in_channels=1):
        super(DQN, self).__init__()

        # For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=2, stride=1)
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(16 * 4, 32, bias=False)
        self.fc5 = nn.Linear(32, num_actions, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(), -1)))
        return self.fc5(x)


class DQNSolver():
    def __init__(self, observation_space, num_actions):
        self.episode_rewards = []
        self.episode_actions = torch.Tensor([])

        net = DQN(observation_space, num_actions)
        print(net)
        params = list(net.parameters())
        print(len(params))
        print(params[0].size())

        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA

        # Episode solver and reward history
        self.solver_history = Variable(torch.Tensor())

        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.episode_actions = torch.Tensor([])

    def remember_reward(self, reward):
        self.episode_rewards.append(reward)

    def predict(self, observation):
        # Select an action (0 or 1) by running solver model and choosing based
        # on the probabilities in observation.
        observation = torch.from_numpy(observation).type(torch.FloatTensor)
        observation = observation.view(2,2)
        action_probs = self.net(observation)
        print(observation)
        print(observation.size())
        quit(0)
        distribution = Categorical(action_probs)
        action = distribution.sample()

        # Add log probability of our chosen action to our history.
        self.episode_actions = torch.cat([
            self.episode_actions,
            distribution.log_prob(action).reshape(1)
        ])

        return action

    def update(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma.
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards.
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / \
                  (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss.
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1), -1)

        # Update model to minimize loss.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()


# class DQNSolver:
#
#     def __init__(self, observation_space, action_space):
#         self.exploration_rate = EXPLORATION_MAX
#
#         self.action_space = action_space
#         self.memory = deque(maxlen=MEMORY_SIZE)
#
#         self.net = Sequential()
#         self.net.add(Dense(24, input_shape=(observation_space,), activation="relu"))
#         self.net.add(Dense(24, activation="relu"))
#         self.net.add(Dense(self.action_space, activation="linear"))
#         self.net.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
#
#     def remember(self, observation, action, reward, next_observation, done):
#         self.memory.append((observation, action, reward, next_observation, done))
#
#     def act(self, observation):
#         if np.random.rand() < self.exploration_rate:
#             return random.randrange(self.action_space)
#         q_values = self.net.predict(observation)
#         return np.argmax(q_values[0])
#
#     def experience_replay(self):
#         if len(self.memory) < BATCH_SIZE:
#             return
#         batch = random.sample(self.memory, BATCH_SIZE)
#         for observation, action, reward, observation_next, terminal in batch:
#             q_update = reward
#             if not terminal:
#                 q_update = (reward + GAMMA * np.amax(self.net.predict(observation_next)[0]))
#             q_values = self.net.predict(observation)
#             q_values[0][action] = q_update
#             self.net.fit(observation, q_values, verbose=0)
#         self.exploration_rate *= EXPLORATION_DECAY
#         self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cart_pole():
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 5000
    # print(env.spec)
    # quit(1)

    # print(env.observation_space)
    # print(env.action_space)

    # 0: Cart position (-4.8 ~ 4.8)
    # 1: Cart velocity
    # 2: Pole angle (-24 ~ 24deg)
    # 3: Pole tip velocity
    observation_space = env.observation_space.shape[0]

    # 0: Push cart to the left
    # 1: Push cart to the right
    num_actions = env.action_space.n

    solver = DQNSolver(observation_space, num_actions)

    run = 0
    dimmed_score = 0
    while run < CYCLES:
        # Reset environment and record the starting observation.
        observation = env.reset()
        step = 0

        while step < 30000:
            if dimmed_score > 500:
                env.render()

            # Pick action.
            action = solver.predict(observation)

            # Step through environment using chosen action.
            observation, reward, done, info = env.step(action.item())

            # observation (4) -> action (1) -> reward (1) -> next observation (4)

            # Save reward.
            solver.remember_reward(reward)

            if done:
                score = step + 1
                dimmed_score = 0.8 * dimmed_score + 0.2 * score
                print(f"Run: {run}, score: {score}, dimmed_score: {dimmed_score}")
                SCORES.append(step)
                break

            step += 1

        # Update solver after each run to do learning.
        solver.update()
        run += 1

        # # Used to determine when the environment is solved.
        # running_reward = (running_reward * 0.99) + (time * 0.01)
        # if episode % 50 == 0:
        #     print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
        #     if running_reward > env.spec.reward_threshold:
        #         print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(
        #             running_reward, time))
        #     break

    # dqn_solver = DQNSolverTorch(observation_space, action_space)
    # run = 0
    # while run < CYCLES:
    #     run += 1
    #     observation = env.reset()
    #     observation = np.reshape(observation, [1, observation_space])
    #     step = 0
    #     while True:
    #         step += 1
    #         # env.render()
    #         action = dqn_solver.act(observation)
    #         observation_next, reward, terminal, info = env.step(action)
    #         reward = reward if not terminal else -reward
    #         observation_next = np.reshape(observation_next, [1, observation_space])
    #         dqn_solver.remember(observation, action, reward, observation_next, terminal)
    #         observation = observation_next
    #         if terminal:
    #             print(f"Run: {run}, exploration: {dqn_solver.exploration_rate}, score: {step}")
    #             SCORES.append(step)
    #             break
    #         dqn_solver.experience_replay()


if __name__ == "__main__":
    cart_pole()

#
# env.reset()
# for i_episode in range(200):
#     observation = env.reset()
#     print(observation)
#
#     for t in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(observation)
#         if done:
#             print(f"Episode {i_episode} finished after {t + 1} timesteps")
#             break
# env.close()

#
# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#     print('ipython')
# else:
#     print('not ipython')
#
# plt.ion()
#
# # if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# plt.show()
