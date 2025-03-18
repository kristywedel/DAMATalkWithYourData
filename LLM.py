import os
import requests
from tqdm import tqdm

# Define the model directory and file path
MODEL_DIR = "./models/models--TheBloke--Llama-2-7B-Chat-GGUF"
MODEL_FILE = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Full path to the model file
model_path = os.path.join(MODEL_DIR, MODEL_FILE)

# Check if the model file already exists
if os.path.exists(model_path):
    print(f"Model file already exists: {model_path}")
else:
    print(f"Downloading model from {MODEL_URL} ...")

    # Download the file with a progress bar
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB

    with open(model_path, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            file.write(chunk)
            bar.update(len(chunk))

    print(f"Model successfully downloaded to: {model_path}")

# Verify the model file exists after download
if os.path.exists(model_path):
    print(f"Model is ready for use at: {model_path}")
else:
    raise FileNotFoundError(f"Download failed. Model file not found: {model_path}")
