import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MiningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(MiningEnv, self).__init__()
        # Observation: temperature, vibration, gas
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        # Actions: 0, 1, 2 (example)
        self.action_space = spaces.Discrete(3)

        # Initial state
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize with random sensor values
        self.state = np.random.rand(3) * 10
        return self.state, {}

    def step(self, action):
        # Randomly update sensor readings for simulation
        self.state += (np.random.rand(3) - 0.5) * 10
        # Example reward: sum of all sensors
        reward = np.sum(self.state)
        done = False  # or implement your logic
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

    def get_sensors(self):
        return {
            "temperature": self.state[0],
            "vibration": self.state[1],
            "gas": self.state[2]
        }

    def get_image(self):
        # Simulated camera feed (random image)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img
