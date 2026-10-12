import os

# Paths
raw_path = os.path.join("data", "raw")
processed_path = os.path.join("data", "processed")

print("✅ Project folders are set up!")
print(f"Raw data folder: {os.path.abspath(raw_path)}")
print(f"Processed data folder: {os.path.abspath(processed_path)}")

# Test write
test_file = os.path.join(processed_path, "test.txt")
with open(test_file, "w") as f:
    f.write("This is a test file inside processed data folder.")

print(f"✅ Wrote a test file: {test_file}")
