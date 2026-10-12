import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mining_env import MiningEnv

# Load trained model
model = PPO.load("models/mining_agent.zip")

# Create environment
env = MiningEnv()

# Initialize plot for rewards
plt.ion()
fig, ax = plt.subplots()
reward_history = []
steps = []

# Reset environment
obs, _ = env.reset()
done = False
step = 0
cumulative_reward = 0

while not done and step < 50:  # run for 50 steps max
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    cumulative_reward += reward
    reward_history.append(cumulative_reward)
    steps.append(step + 1)
    
    print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Cumulative: {cumulative_reward:.2f}")
    
    # Update plot
    ax.clear()
    ax.plot(steps, reward_history, marker='o')
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Mining Agent Simulation")
    plt.pause(0.1)
    
    step += 1
    time.sleep(0.2)  # slow down for visualization

plt.ioff()
plt.show()

