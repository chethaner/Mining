from stable_baselines3 import PPO
from mining_env import MiningEnv

# Load the trained model
model = PPO.load("models/mining_agent")  # no need to add .zip

# Create environment
env = MiningEnv()
obs, _ = env.reset()

for step in range(10):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1} | Action: {action} | Reward: {reward}")
    
    if terminated or truncated:
        obs, _ = env.reset()
