from mining_env import MiningEnv  # imports the MiningEnv class

# Create an instance of the environment
env = MiningEnv()

# Reset environment to get the initial observation
obs, _ = env.reset()
print("Observation shape:", obs.shape)  # Shows the size of input data

# Take one action (1 in this case) in the environment
obs, reward, done, truncated, info = env.step(1)
print("Next obs shape:", obs.shape, "Reward:", reward)  # Shows next observation and reward
