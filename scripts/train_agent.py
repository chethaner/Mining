from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mining_env import MiningEnv
import os

# --- Create environment ---
env = DummyVecEnv([lambda: MiningEnv()])

# --- Create folder for models ---
os.makedirs("models", exist_ok=True)

# --- Initialize PPO agent ---
model = PPO(
    "MlpPolicy",   # Multi-Layer Perceptron for non-image inputs
    env,
    verbose=1,
    tensorboard_log="logs/"
)

# --- Train the agent ---
model.learn(total_timesteps=5000)

# --- Save the model ---
model.save("models/mining_agent")
print("✅ Training completed and model saved!")
