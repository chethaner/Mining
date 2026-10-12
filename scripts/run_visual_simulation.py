import time
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mining_env import MiningEnv

# Load trained model
model = PPO.load("models/mining_agent.zip")

# Create environment
env = MiningEnv()

# Reset environment
obs, _ = env.reset()
done = False
step = 0
cumulative_reward = 0
reward_history = []
steps = []

# Setup matplotlib for reward plotting
plt.ion()
fig, ax = plt.subplots()

while not done and step < 50:
    # Agent predicts next action
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    cumulative_reward += reward
    reward_history.append(cumulative_reward)
    steps.append(step + 1)
    
    print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Cumulative: {cumulative_reward:.2f}")
    
    # --- Visualize sensor data ---
    # Assuming env has a method to get current sensor readings and an image
    try:
        sensors = env.get_sensors()  # dict: {"temperature": ..., "vibration": ..., "gas": ...}
        img = env.get_image()        # OpenCV image
    except AttributeError:
        sensors = {"temperature": 0, "vibration": 0, "gas": 0}
        img = None

    # Plot cumulative reward
    ax.clear()
    ax.plot(steps, reward_history, marker='o')
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Mining Agent Simulation")
    plt.pause(0.1)
    
    # Show image if exists
    if img is not None:
        cv2.imshow("Mining Camera Feed", img)
    
    # Print sensor readings
    print(f"Sensors: Temp={sensors['temperature']}, Vibration={sensors['vibration']}, Gas={sensors['gas']}")
    
    step += 1
    time.sleep(0.2)

plt.ioff()
cv2.destroyAllWindows()
plt.show()
