import os
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

USE_HF_API = os.getenv("USE_HF_API", "0") == "1"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_REPO = os.getenv("HF_API_REPO", "Salesforce/blip-image-captioning-base")

def get_model(device: str = "cpu"):
    """
    Returns a model-like object.
    If USE_HF_API is set, returns HFInferenceWrapper that talks to the Hugging Face Serverless Inference API.
    Otherwise returns a simple dummy model suitable for training (random weights).
    """
    if USE_HF_API:
        from .hf_inference import HFInferenceWrapper
        return HFInferenceWrapper(
            repo_id=HF_API_REPO,
            token=HF_API_TOKEN,
            device=device,
        )
    else:
        # Dummy model for training: we just need something that produces logits of shape (B, seq_len, vocab_size)
        # We'll mimic the interface expected by train_dummy.py: forward(images, input_ids, attention_mask) -> logits
        class DummyCaptionModel(nn.Module):
            def __init__(self, vocab_size=50257, max_len=30, dim=768):
                super().__init__()
                self.vocab_size = vocab_size
                self.max_len = max_len
                # a simple linear layer that maps image features to logits; we'll ignore actual image input
                self.dummy = nn.Linear(dim, vocab_size)
                # we also need to produce prefix-like outputs for generate (not used in training)
                self.prefix_len = 10
                self.prefix_dim = dim

            def forward(self, images, input_ids, attention_mask):
                """
                images: tensor or list (unused)
                input_ids: (B, seq_len)
                attention_mask: (B, seq_len)
                Returns logits of shape (B, seq_len, vocab_size)
                """
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                # produce random logits
                logits = torch.randn(batch_size, seq_len, self.vocab_size, requires_grad=True)
                return logits.to(images.device if hasattr(images, 'device') else torch.device('cpu'))

            @torch.no_grad()
            def generate(self, images, max_length=30, num_beams=1, temperature=1.0, top_p=0.9, do_sample=True, **kwargs):
                """
                Dummy generation: returns fixed token ids (e.g., repeats of eos token) to keep compatibility.
                In HF API mode we won't call this.
                """
                batch_size = images.shape[0] if torch.is_tensor(images) else len(images)
                # Just return eos token repeated
                eos_token_id = 50256  # GPT-2 eos
                generated = torch.full((batch_size, 1), eos_token_id, dtype=torch.long, device=images.device if torch.is_tensor(images) else torch.device('cpu'))
                return generated

        return DummyCaptionModel().to(device)