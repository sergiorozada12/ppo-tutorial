from typing import List, Tuple

from src.environments import CustomPendulumEnv
from src.agents import PPOAgent


def train(
    env: CustomPendulumEnv,
    ppo_agent: PPOAgent,
    epochs: int,
    max_steps: int,
    update_freq: int
) -> Tuple[PPOAgent, List[float]]:

    returns = []
    for epoch in range(epochs):
        state = env.reset()
        cum_reward = 0

        for step in range(max_steps):
            action = ppo_agent.select_action(state)
            state_next, reward, done, _ = env.step(action)
            
            ppo_agent.buffer.states_next.append(state_next)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.terminals.append(done)
            cum_reward += reward
            
            if len(ppo_agent.buffer) == update_freq:
                ppo_agent.update()

            if done:
                break
                
            state = state_next
        
        returns.append(cum_reward)
        print(f'{epoch}/{epochs}: {returns[-1]} \r', end='')
    return ppo_agent, returns
