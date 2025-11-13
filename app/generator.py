# app/generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Safety: reduce parallelism inside Python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

from app.config import GENERATION_MODEL

class LocalGenerator:
    """
    Wrapper for seq2seq model (FLAN-T5). Uses CPU by default; single-threaded.
    """
    def __init__(self, model_name: str = GENERATION_MODEL):
        # Ensure model_name is resolved (from .env) before loading
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # move model to device explicitly
        self.model.to(self.device)
        # Reduce torch intra-op threads (extra guard)
        torch.set_num_threads(1)

    def __call__(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
