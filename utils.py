from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import json
import glob
import h5py
from typing import Tuple
import os
import torch

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Check if model files are directly in the path, or in a subdirectory
    if os.path.exists(os.path.join(model_path, "config.json")) and os.path.exists(os.path.join(model_path, "tokenizer.json")):
        actual_model_path = model_path
    else:
        # Look for subdirectories containing the model files
        if os.path.exists(model_path):
            subdirs = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
            for subdir in subdirs:
                subdir_path = os.path.join(model_path, subdir)
                if os.path.exists(os.path.join(subdir_path, "config.json")) and os.path.exists(os.path.join(subdir_path, "tokenizer.json")):
                    actual_model_path = subdir_path
                    print(f"Found model files in subdirectory: {actual_model_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find model files in {model_path} or its subdirectories")
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist")

    model_path = actual_model_path

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        assert tokenizer.padding_side == "right"
    except Exception as e:
        print(f"AutoTokenizer failed, trying to load tokenizer.json directly: {e}")
        # Try to load tokenizer.json directly
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, padding_side="right")
        else:
            raise RuntimeError(f"Could not find tokenizer files in {model_path}")

    # Load the .h5 weights file
    h5_file = os.path.join(model_path, "model.weights.h5")
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"model.weights.h5 not found in {model_path}")

    # Load all tensors from the h5 file
    tensors = {}
    with h5py.File(h5_file, 'r') as f:
        for key in f.keys():
            # Convert numpy arrays to torch tensors
            tensors[key] = torch.from_numpy(f[key][()])

    # Load the model's config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)