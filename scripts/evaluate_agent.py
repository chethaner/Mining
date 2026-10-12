import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mining_env import MiningEnv

# Load the trained model
model = PPO.load("models/mining_agent.zip")

# Create the environment
env = MiningEnv()

episodes = 5
all_rewards = []

for ep in range(episodes):
    obs, _ = env.reset()
    ep_rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        ep_rewards.append(reward)
    all_rewards.append(sum(ep_rewards))
    print(f"Episode {ep+1} | Total Reward: {sum(ep_rewards)}")

# Plot rewards
plt.plot(all_rewards, marker='o')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Agent Performance")
plt.show()
