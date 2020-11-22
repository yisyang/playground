import gym
import numpy as np
from os import path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

SCORES = []
RENDER_ON_SOLVED = True
MAX_STEPS = 1000
SOLVED_SCORE = 996

CYCLES = 10000

GAMMA = 0.98                # Amount to discount future rewards.
LEARNING_RATE = 0.004       # Speed at which weight changes.

MODEL_SAVE_PATH = './net/cart-pole-linear-net.pth'
MODEL_LOAD_PATH = './net/cart-pole-linear-net.pth'


# noinspection PyPep8Naming,DuplicatedCode
class DQN(nn.Module):
    def __init__(self, observation_space, num_actions):
        super(DQN, self).__init__()
        self.fc3 = nn.Linear(observation_space, 48, bias=False)
        self.do4 = nn.Dropout(p=0.3)
        self.fc5 = nn.Linear(48, num_actions, bias=False)
        self.sm6 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc3(x)
        x = F.relu(self.do4(x))
        x = self.fc5(x)
        x = self.sm6(x)

        return x


# noinspection PyPep8Naming,DuplicatedCode
class DQNSolver:
    def __init__(self, observation_space, num_actions):
        self.episode_rewards = []
        self.episode_actions = torch.Tensor([])

        net = DQN(observation_space, num_actions)

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
        action_probabilities = self.net(observation)
        distribution = Categorical(action_probabilities)
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
        # (t-i)
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / \
                  (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss.
        # sum(-1 * (t-i) * prob)
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1))

        # Update network weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def save(self):
        torch.save(self.net.state_dict(), MODEL_SAVE_PATH)
        print(f'Net saved at {MODEL_SAVE_PATH}.')

    def load(self):
        if path.exists(MODEL_LOAD_PATH):
            self.net.load_state_dict(torch.load(MODEL_LOAD_PATH))
            print(f'Net loaded from {MODEL_LOAD_PATH}.')
        else:
            print('Saved net data does not exist.')


def cart_pole():
    env = gym.make('CartPole-v1')
    env._max_episode_steps = MAX_STEPS

    # 0: Cart position (-4.8 ~ 4.8)
    # 1: Cart velocity
    # 2: Pole angle (-24 ~ 24deg)
    # 3: Pole tip velocity
    observation_space = env.observation_space.shape[0]

    # 0: Push cart to the left
    # 1: Push cart to the right
    num_actions = env.action_space.n

    solver = DQNSolver(observation_space, num_actions)
    solver.load()

    run = 0
    solved = False
    dimmed_score = 0
    while run < CYCLES and not solved:
        # Reset environment and record the starting observation.
        observation = env.reset()
        step = 0

        if dimmed_score > SOLVED_SCORE:
            print(f"Solved on run #{run}.")
            solved = True
            solver.save()

        while step < MAX_STEPS:
            if solved and RENDER_ON_SOLVED:
                env.render()

            # Pick action.
            action = solver.predict(observation)

            # Step through environment using chosen action.
            observation, reward, done, info = env.step(action.item())

            # Hack reward to encourage net to:
            # 1: Stay alive.
            # 2: Stay near the center.
            reward = reward - abs(observation[0])/4.8

            # Save reward.
            solver.remember_reward(reward)

            step_next = step + 1
            if done or step_next == MAX_STEPS:
                dimmed_score = 0.8 * dimmed_score + 0.2 * step_next
                print(f"Run: {run}, score: {step_next}, dimmed_score: {dimmed_score}")
                SCORES.append(step)
                break

            step += 1

        # Update solver after each run to do learning.
        solver.update()
        run += 1


if __name__ == "__main__":
    cart_pole()
