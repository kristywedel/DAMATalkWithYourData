from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM

# Download Llama 2 GGUF model
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

print(f"Model downloaded to: {model_path}")

# Load and run the model
model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")

# Example query
query = "Explain quantum computing in simple terms."
response = model(query, max_new_tokens=150)

print("\nChatbot Response:")
print(response)
