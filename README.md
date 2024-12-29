# Casual Attention with Padded Inputs via PyTorch FlexAttention

This tutorial script covers how to handle both causal attention and padded inputs with the new FlexAttention and BlockMask features of torch >= 2.5.

I was unable to find any clear code or discussions online on covering padded input sequences and FlexAttention, so I thought I'd describe one of implementing it along with causal attention. 

I will not be going over the details of FlexAttention, but check out PyTorch's <a href="https://pytorch.org/blog/flexattention/ ">blog</a> if you are curious.

### Install:
Here we install via the <a href="https://github.com/pytorch-labs/attention-gym">attention-gym github</a> as it will ensure compatibility and give us access to their visualization tool. 

```
git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ../
```

## Breakdown of [<u>causal_padded_flexattn.py</u>](./causal_padded_flexattn.py):

### Imports

```python
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask


# For visualizing attention scores after masking
import sys
sys.path.append("./attention-gym")
from attn_gym import visualize_attention_scores
from pathlib import Path
```

### MultiheadFlexAttention

```python
class MultiheadFlexAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, bias=False):
        """
        description: a torch module that implements multiheaded self-attention via flex_attention.
        args:
            d_in: int, the dimension of the input tensor.
            d_out: int, the dimension of the output tensor.
            n_heads: int, the number of heads to use for the multiheaded self-attention.
            bias: bool, whether to use query, key, and value biases
        """
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.d_out = d_out

        self.in_proj = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x, block_mask):
        """
        description: forward pass of the multiheaded self-attention module.
        args:
            x: torch.Tensor, the input tensor of shape (batch_size, seq_len, d_in)
            block_mask: torch.Tensor, the block mask to use for flex_attention
        """
        batch_size, seq_len, d_in = x.shape

        # Create stacked qkv via input projection
        qkv = self.in_proj(x) # (batch_size, seq_len , 3 * d_in)

        # Split qkv and divide d_in into heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head) # (batch_size, seq_len, 3, n_heads, d_head)

        # Permute shape of qkv for flex_attention
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, seq_len, d_head)

        # Get queries, keys, values
        queries, keys, values = qkv # 3 x (batch_size, n_heads, seq_len, d_head)

        # Calculate attention via flex_attention
        attn = flex_attention(queries, keys, values, block_mask=block_mask) # (batch_size, n_heads, seq_len, d_head)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        # Pass attention output through output projection
        attn = self.out_proj(attn)

        return attn, queries, keys
```


## Resources that helped me 
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention 
- https://pytorch.org/blog/flexattention/ 
- https://github.com/ViktorooReps/llm-experiments/blob/59fe19a6fe6be2cd3652f75afcc90156953889cc/src/models/modelling_llama_long_context.py#L226
- https://github.com/pytorch-labs/attention-gym/issues/38