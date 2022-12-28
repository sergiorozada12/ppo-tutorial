from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils import Buffer


class Agent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Agent, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).double()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).double()
        
        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)

    def pi(self, state: np.ndarray) -> torch.distributions.MultivariateNormal:
        state = torch.as_tensor(state).double()
        
        # Parameters
        mu = self.actor(state)
        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        
        # Distribution
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action.reshape(-1, 1))
        return action_logprob
    
    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        value = self.critic(state)
        return value

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        epochs: int,
        eps_clip: float
    ) -> None:

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        
        self.buffer = Buffer()

        self.policy = Agent(state_dim, action_dim)
        self.policy_old = Agent(state_dim, action_dim)
        
        mu_params = list(self.policy.actor.parameters())
        std_params = [self.policy.log_sigma]
        
        self.opt_actor = torch.optim.Adam(mu_params + std_params, lr_actor)
        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr_critic)

        self.policy_old.load_state_dict(self.policy.state_dict())        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.numpy()

    def update(self) -> None:
        rewards = torch.as_tensor(rewards).double()

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_states_next = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()

        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions)
            state_values = self.policy.evaluate_value(old_states)
            state_next_values = self.policy.evaluate_value(old_states_next)
            
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            td_target = rewards + self.gamma*state_next_values 
            td_estimation = state_values
            td_error = td_target.detach() - td_estimation.detach()
            
            minorizer_raw = ratio * td_error
            minorizer_clamped = ratio_clamped * td_error

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped) 
            loss_critic = self.MseLoss(td_target.detach(), td_estimation)
            
            # Actor
            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.opt_actor.step()
            
            # Critic
            self.opt_critic.zero_grad()
            loss_critic.mean().backward()
            self.opt_critic.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
