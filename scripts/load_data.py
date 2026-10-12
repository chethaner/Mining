import os
import pandas as pd
import cv2

# --- Sensor Data ---
sensor_file = os.path.join("data", "raw", "sensors", "mining_sensors.csv")
if os.path.exists(sensor_file):
    df = pd.read_csv(sensor_file)
    print("✅ Loaded sensor data:")
    print(df.head())
else:
    print("⚠️ No sensor file found.")

# --- Image Data ---
image_dir = os.path.join("data", "raw", "images")
if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
    sample_img = os.listdir(image_dir)[0]
    img_path = os.path.join(image_dir, sample_img)
    img = cv2.imread(img_path)
    if img is not None:
        print(f"✅ Loaded image: {sample_img}, shape={img.shape}")
    else:
        print("⚠️ Could not read image file.")
else:
    print("⚠️ No images found.")
