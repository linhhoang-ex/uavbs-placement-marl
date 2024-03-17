"""Basic code which shows what it's like to run PPO on the Pistonball env using
the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer
to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import os
import datetime
from envs.sagin_v1 import *
from envs.utils import *


class Agent(nn.Module):
    def __init__(self, obs_len, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(obs_len, 1024)),
            nn.ReLU(),
            self._layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 1.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 1.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)

    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)

    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float64)
    writer = SummaryWriter()

    """ALGO PARAMS"""
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    # stack_size = 1
    # frame_size = (64, 64)
    max_cycles = 128
    total_episodes = 10000
    lrate = 1e-3
    n_reuse = 3             # no. of reuse times for the experiment of an episode
    seed = None

    """ ENV SETUP """
    env = parallel_env(seed=seed, max_cycles=max_cycles)
    # env = color_reduction_v0(env)
    # env = resize_v1(env, frame_size[0], frame_size[1])
    # env = frame_stack_v1(env, stack_size=stack_size)
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    obs_shape = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(obs_len=obs_shape[0], num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lrate, eps=lrate / 100)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    # rb_obs = torch.zeros((max_cycles, num_agents, stack_size, *frame_size)).to(device)
    rb_obs = torch.zeros((max_cycles, num_agents, obs_shape[0])).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    log_ep_return = np.full(shape=(total_episodes,), fill_value=np.nan)
    log_value_loss = np.full(shape=(total_episodes,), fill_value=np.nan)
    log_policy_loss = np.full(shape=(total_episodes,), fill_value=np.nan)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in tqdm(range(total_episodes), file=sys.stdout):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset()
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            end_step = max_cycles - 1       # if not terminated early
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step     # terminated early
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t] + gamma * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(n_reuse):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                # clip fraction
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        env.close()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_ep_return[episode] = np.mean(total_episodic_return)
        log_value_loss[episode] = v_loss.item()
        log_policy_loss[episode] = pg_loss.item()

        # tqdm.write(f"Training episode {episode}")
        # tqdm.write(f"Episodic Return: {log_ep_return[episode]}")
        # rwd = 50
        # if episode >= rwd:
        #     mavg_return = np.mean(log_ep_return[episode - rwd + 1: episode + 1])
        #     tqdm.write(
        #         f"Ep. Return (MAVG) ({rwd}): {mavg_return}"
        #     )
        # tqdm.write(f"Episode Length: {end_step}")
        # tqdm.write("")
        # tqdm.write(f"Value Loss: {log_value_loss[episode]}")
        # tqdm.write(f"Policy Loss: {log_policy_loss[episode]}")
        # tqdm.write(f"Old Approx KL: {old_approx_kl.item()}")
        # tqdm.write(f"Approx KL: {approx_kl.item()}")
        # tqdm.write(f"Clip Fraction: {np.mean(clip_fracs)}")
        # tqdm.write(f"Explained Variance: {explained_var.item()}")
        # tqdm.write("\n-------------------------------------------\n")

        writer.add_scalar("charts/episodic_return", log_ep_return[episode], episode)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], episode
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), episode)
        writer.add_scalar("losses/explained_variance", explained_var, episode)

    writer.flush()
    writer.close()

    """ SAVE THE TRAINED MODEL """
    # Ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    PATH = "trained_agents/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(PATH)

    metadata = {
        "name": "ppo_sagin_v1_1",
        'episode': total_episodes,
    }
    with open(os.path.join(PATH, "metadata.txt"), 'w') as f:
        for key, value in metadata.items():
            f.write('%s: %s\n' % (key, value))

    model = agent.network
    torch.save({
        'episode': total_episodes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'log_ep_return': log_ep_return,
        'log_value_loss': log_value_loss,
        'log_policy_loss': log_policy_loss,
        "metadata": metadata
    }, PATH + "/model.tar")

    print(f"Saved the trained agent to {PATH}")
