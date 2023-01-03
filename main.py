import matplotlib.pyplot as plt

from src.environments import CustomPendulumEnv
from src.agents import PPOAgent
from src.algorithms import train


STATE_DIM = 2           # dimensions of the state space
ACTION_DIM = 1          # dimensions of the action space
EPOCHS = 10_000         # max number of episodes
MAX_STEPS = 100         # max timesteps in one episode
UPDATE_FREQ = 40        # steps frequency to update
TRAINING_EPOCHS = 40    # update policy for K epochs in one PPO update
EPS_CLIP = 0.2          # clip parameter for PPO
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-5         # learning rate for actor network
LR_CRITIC = 1e-4        # learning rate for critic network


if __name__ == "__main__":

    env = CustomPendulumEnv()
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        epochs=TRAINING_EPOCHS,
        eps_clip=EPS_CLIP
    )

    agent, returns = train(
        env=env,
        ppo_agent=agent,
        epochs=EPOCHS,
        max_steps=MAX_STEPS,
        update_freq=UPDATE_FREQ
    )

    fig = plt.figure(figsize=[8, 7])
    plt.grid()
    plt.plot(returns)
    plt.xlim(0, 10_000)
    plt.ylim(0, 110)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.tight_layout()
    fig.savefig('figures/fig_results.jpg', dpi=300)
