import numpy as np
import wave

# Parameters
frequency = 1000  # Hz
duration = 0.3    # seconds
sample_rate = 44100  # samples per second

# Generate sine wave
t = np.linspace(0, duration, int(sample_rate * duration), False)
tone = np.sin(frequency * t * 2 * np.pi)

# Ensure it's in 16-bit PCM format
audio = (tone * 32767).astype(np.int16)

# Save to WAV file
with wave.open("alert.wav", "w") as f:
    f.setnchannels(1)          # mono
    f.setsampwidth(2)          # 2 bytes per sample
    f.setframerate(sample_rate)
    f.writeframes(audio.tobytes())

print("alert.wav created successfully!")
