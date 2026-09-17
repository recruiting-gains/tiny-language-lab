"""An original, small pre-norm causal decoder. No pretrained weights."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass
class Config:
    vocab_size: int
    context_length: int = 48
    embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2

    def __post_init__(self):
        if not 2 <= self.vocab_size <= 256 or not 4 <= self.context_length <= 128:
            raise ValueError("Vocabulary/context outside tiny-model limits.")
        if not 8 <= self.embed_dim <= 128 or not 1 <= self.num_layers <= 4:
            raise ValueError("Model size outside tiny-model limits.")
        if self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError("Embedding size must divide evenly into heads.")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.heads = config.num_heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        reshape = lambda z: z.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim // self.heads)
        # Position i cannot inspect j > i. This is the causal mask.
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        attention = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        mixed = (attention @ v).transpose(1, 2).contiguous().view(batch, length, dim)
        x = x + self.proj(mixed)
        return x + self.fc2(torch.relu(self.fc1(self.ln2(x)))), attention


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(config.embed_dim, eps=1e-5)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, inspect=False):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context_length:
            raise ValueError("Supply a nonempty batch within the context length.")
        if ids.dtype != torch.long or torch.any(ids < 0) or torch.any(ids >= self.config.vocab_size):
            raise ValueError("Token IDs must be vocabulary indices with integer dtype.")
        x = self.token_embedding(ids) + self.position_embedding(torch.arange(ids.shape[1], device=ids.device))
        attention = None
        for block in self.blocks:
            x, attention = block(x)
        logits = self.lm_head(self.ln_final(x))
        return (logits, attention) if inspect else logits

    @torch.no_grad()
    def generate(self, tokenizer, prompt="the ", count=80, temperature=0.8, seed=31):
        if not prompt or len(prompt) > 512:
            raise ValueError("Prompt must contain 1–512 characters.")
        if type(count) is not int or not 0 <= count <= 256:
            raise ValueError("Generate between 0 and 256 characters.")
        if not math.isfinite(temperature) or not 0.2 <= temperature <= 1.5:
            raise ValueError("Temperature must be from 0.2 to 1.5.")
        self.eval()
        ids = tokenizer.encode(prompt)
        rng = torch.Generator().manual_seed(seed)
        for _ in range(count):
            context = torch.tensor([ids[-self.config.context_length:]], dtype=torch.long)
            probabilities = (self(context)[0, -1] / temperature).softmax(-1)
            ids.append(int(torch.multinomial(probabilities, 1, generator=rng)))
        return tokenizer.decode(ids)

    def config_dict(self):
        return asdict(self.config)
