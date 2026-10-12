import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from stable_baselines3 import PPO
from mining_env import MiningEnv
import cv2
import simpleaudio as sa
import csv
from datetime import datetime

# --- Thresholds for alerts ---
TEMP_THRESHOLD = 50
VIB_THRESHOLD = 1.0
GAS_THRESHOLD = 600

# --- Create folders ---
os.makedirs("alerts", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_file = "logs/sensor_log.csv"

# --- Initialize CSV log ---
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Step", "Temperature", "Vibration", "Gas", "CumulativeReward"])

# --- Load trained model ---
model = PPO.load("models/mining_agent.zip")

# --- Create environment ---
env = MiningEnv()
obs, _ = env.reset()
done = False

# --- Data storage ---
step = 0
cumulative_reward = 0
reward_history = []
steps_list = []
temp_history = []
vib_history = []
gas_history = []

# --- Setup dashboard ---
plt.ion()
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(3, 2, figure=fig)

ax_camera = fig.add_subplot(gs[:, 0])
ax_reward = fig.add_subplot(gs[0, 1])
ax_temp = fig.add_subplot(gs[1, 1])
ax_vib = fig.add_subplot(gs[2, 1])

while not done and step < 500:  # Run longer to see trends
    # --- AI action ---
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

    cumulative_reward += reward
    reward_history.append(cumulative_reward)
    steps_list.append(step + 1)

    # --- Get sensors and image ---
    try:
        sensors = env.get_sensors()
        img = env.get_image()
    except AttributeError:
        # Simulated fallback
        sensors = {
            "temperature": np.random.randint(20, 70),
            "vibration": np.random.random()*2,
            "gas": np.random.randint(100, 800)
        }
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    temp_history.append(sensors["temperature"])
    vib_history.append(sensors["vibration"])
    gas_history.append(sensors["gas"])

    # --- Check alerts ---
    alert = False
    if sensors["temperature"] > TEMP_THRESHOLD or \
       sensors["vibration"] > VIB_THRESHOLD or \
       sensors["gas"] > GAS_THRESHOLD:
        alert = True
        try:
            wave_obj = sa.WaveObject.from_wave_file("scripts/alert.wav")
            play_obj = wave_obj.play()
        except FileNotFoundError:
            print("Alert WAV file missing! Check scripts/alert.wav")

        # Save snapshot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"alerts/alert_{timestamp}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # --- Camera feed ---
    ax_camera.clear()
    ax_camera.imshow(img)
    if alert:
        for spine in ax_camera.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(5)
    else:
        for spine in ax_camera.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
    ax_camera.set_title("Mining Camera Feed")
    ax_camera.axis("off")

    # --- Cumulative Reward plot ---
    ax_reward.clear()
    ax_reward.plot(steps_list, reward_history, marker='o', color='orange')
    ax_reward.set_title("Cumulative Reward")

    # --- Temperature line plot ---
    ax_temp.clear()
    ax_temp.plot(steps_list, temp_history, color='blue', label="Temperature")
    ax_temp.axhline(y=TEMP_THRESHOLD, color='red', linestyle='--', label="Temp Threshold")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_title("Temperature Over Time")
    ax_temp.legend()

    # --- Vibration & Gas line plot ---
    ax_vib.clear()
    ax_vib.plot(steps_list, vib_history, color='green', label="Vibration")
    ax_vib.plot(steps_list, gas_history, color='purple', label="Gas Level")
    ax_vib.axhline(y=VIB_THRESHOLD, color='red', linestyle='--', label="Vib Threshold")
    ax_vib.axhline(y=GAS_THRESHOLD, color='red', linestyle=':', label="Gas Threshold")
    ax_vib.set_ylabel("Vibration / Gas")
    ax_vib.set_xlabel("Step")
    ax_vib.set_title("Vibration & Gas Over Time")
    ax_vib.legend()

    # --- Log sensor readings to CSV ---
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, sensors["temperature"], sensors["vibration"], sensors["gas"], cumulative_reward])

    plt.pause(0.1)
    step += 1
    time.sleep(0.2)

plt.ioff()
plt.show()
