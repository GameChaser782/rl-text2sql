# Unsloth MUST be imported FIRST before transformers or torch
try:
    from unsloth import FastLanguageModel

    HAS_UNSLOTH = True
except ImportError as e:
    HAS_UNSLOTH = False
    print(f"WARNING: Unsloth import failed: {e}")

# Now import everything else
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(
    model_name: str,
    use_qlora: bool = True,
    use_unsloth: bool = False,
    num_gpus: int = 1,
    max_seq_length: int = 2048,
):
    """
    Load model and tokenizer with specified configuration.

    Args:
        model_name: HuggingFace model name
        use_qlora: Whether to use QLoRA quantization
        use_unsloth: Whether to use Unsloth optimization (Linux/CUDA only)
        num_gpus: Number of GPUs available (1 or 2)
        max_seq_length: Maximum sequence length for unsloth

    Returns:
        model, tokenizer
    """

    # Determine device map based on GPU count
    if num_gpus == 1:
        device_map = {"": 0}
    else:
        device_map = "auto"

    print(
        f"Loading model {model_name} with use_unsloth={use_unsloth}, use_qlora={use_qlora}, num_gpus={num_gpus}"
    )

    if use_unsloth:
        if not HAS_UNSLOTH:
            raise ImportError(
                "Unsloth is not installed. Please install it with: pip install unsloth[kaggle] or set use_unsloth=False."
            )

        # Load model with Unsloth optimization
        # Unsloth patches the model for faster training with less VRAM
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect dtype
            load_in_4bit=use_qlora,
        )

        # Add LoRA adapters with Unsloth's optimized method
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's efficient gradient checkpointing
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Unsloth tokenizer is already properly configured

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Prepare for k-bit training
            model = prepare_model_for_kbit_training(model)

            # LoRA configuration
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Add LoRA adapters
            model = get_peft_model(model, lora_config)

            print(f"LoRA trainable parameters: {model.print_trainable_parameters()}")

        else:
            # Load model normally (requires more memory)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()

    return model, tokenizer
