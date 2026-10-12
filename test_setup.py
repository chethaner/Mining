import torch
import cv2
import gymnasium as gym

print("✅ Python is working inside neuromine environment")

# Check Torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Check OpenCV
img = 255 * torch.ones((100, 100, 3), dtype=torch.uint8).numpy()
cv2.imwrite("test_image.jpg", img)
print("✅ OpenCV created test_image.jpg")

# Check Gym
try:
    env = gym.make("CartPole-v1")
    obs = env.reset()
    print("✅ Gym environment works:", obs)
    env.close()
except Exception as e:
    print("⚠️ Gym error:", e)
