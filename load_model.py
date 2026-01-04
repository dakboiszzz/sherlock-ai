
import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig


# Create quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",              # Use NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,   # Compute in fp16
    bnb_4bit_use_double_quant=True,         # Extra compression
)
# Create LoRA config
lora_config = LoraConfig(
    r=16,                                    # Rank
    lora_alpha=32,                           # Alpha (2x rank)
    target_modules=[                         # Attention layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading LLaVA model in 4-bit...")
print("(First run downloads ~4GB)")

model_name = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("\n✓ Model loaded!")
print("="*60)

if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {memory_used:.2f} GB")
