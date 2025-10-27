import pandas as pd
import os

# Test different path formats
paths_to_try = [
    '/c/nrl_work/training.csv',
    'C:/nrl_work/training.csv',
    'C:\\nrl_work\\training.csv',
    '/mnt/c/nrl_work/training.csv'
]

for path in paths_to_try:
    print(f"\nTrying: {path}")
    print(f"os.path.exists: {os.path.exists(path)}")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"works")
            print(f"Loaded {len(df)} rows")
            break
        except Exception as e:
            print(f"nah: {e}")