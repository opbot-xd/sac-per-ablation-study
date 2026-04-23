import time, os, sys
import gymnasium as gym
import gym_anm
from io import StringIO

env = gym.make("ANM6Easy-v0")
obs, _ = env.reset(seed=42)

t0 = time.time()
for _ in range(10_000):
    sys.stdout = StringIO()  # suppress gym-anm prints
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    sys.stdout = sys.__stdout__  # restore
    if term or trunc:
        obs, _ = env.reset()
elapsed = time.time() - t0

ms_per_step = elapsed / 10_000 * 1000
total_hours = (ms_per_step / 1000) * 300_000 * 24 / 4 / 3600

print(f"ms per step: {ms_per_step:.2f}")
print(f"Projected total (24 runs, 4 workers): {total_hours:.1f} hours")
env.close()