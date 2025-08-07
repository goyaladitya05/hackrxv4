import torch
from transformers import AutoModel, AutoTokenizer

def check_gpu():
    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    else:
        print("❌ CUDA is NOT available. Running on CPU.")

    # Load small model to test GPU usage
    model_name = "BAAI/bge-base-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model loaded on: {device}")

    # Dummy input
    text = "This is a GPU test."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        print("✅ Forward pass completed.")

if __name__ == "__main__":
    check_gpu()
