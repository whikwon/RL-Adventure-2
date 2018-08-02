import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from osim.env import *
from multiprocessing_env import SubprocVecEnv
from tensorboard_logging import Logger


# Hyperparameters
NUM_ENVS = 24
num_inputs  = 158 
num_outputs = 19
hidden_size = 512
lr = 3e-4
num_steps = 20
mini_batch_size = 96
ppo_epochs = 10
threshold_reward = 2000
max_frames = 1000000

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


def make_env():
    def _thunk():
        env = ProstheticsEnv(False)
        return env

    return _thunk


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(np.clip(action.cpu().numpy()[0], 0, 1))
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


# Model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
	logger = Logger('./log')
	env = ProstheticsEnv(False)
	envs = [make_env() for i in range(NUM_ENVS)]
	envs = SubprocVecEnv(envs)
	model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)

	frame_idx = 0
	test_rewards = []

	state = envs.reset()

	while frame_idx < max_frames:

		log_probs = []
		values    = []
		states    = []
		actions   = []
		rewards   = []
		masks     = []
		entropy = 0

		for _ in range(num_steps):
			state = torch.FloatTensor(state).to(device)
			dist, value = model(state)

			action = dist.sample()
			next_state, reward, done, _ = envs.step(np.clip(action.cpu().numpy(), 0, 1))

			log_prob = dist.log_prob(action)
			entropy += dist.entropy().mean()
			
			log_probs.append(log_prob)
			values.append(value)
			rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
			masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
			
			states.append(state)
			actions.append(action)
			
			state = next_state
			frame_idx += 1
			
			if frame_idx % 1000 == 0:
				test_reward = np.mean([test_env() for _ in range(1)])
				logger.log_scalar('Reward', test_reward, frame_idx)
	#				plot(frame_idx, test_rewards)

		next_state = torch.FloatTensor(next_state).to(device)
		_, next_value = model(next_state)
		returns = compute_gae(next_value, rewards, masks, values)

		returns   = torch.cat(returns).detach()
		log_probs = torch.cat(log_probs).detach()
		values    = torch.cat(values).detach()
		states    = torch.cat(states)
		actions   = torch.cat(actions)
		advantage = returns - values
		ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
	# save model when training ends
	model.save(model, 'model.pth')
