import os
from playsound import playsound

base_dir = r"C:\nrl_work\training"
folder_name = "030"
file_name = "1.wav"

file_path = os.path.join(base_dir, folder_name, file_name)
print(f"Playing {file_path}")

if os.path.exists(file_path):
    playsound(file_path)
else:
    print("File not found.")
