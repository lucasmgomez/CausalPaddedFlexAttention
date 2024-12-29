# Casual Attention with Padded Inputs via PyTorch FlexAttention

This small script covers how to handle both causal attention and padded inputs with the new FlexAttention and BlockMask features of torch >= 2.5.

I was unable to find any clear code or discussions online covering padded input sequences and FlexAttention, so I thought I'd describe one way of implementing it along with causal attention. 

I will not be going over the details of FlexAttention, but check out PyTorch's <a href="https://pytorch.org/blog/flexattention/ ">blog</a> if you are curious.

### Install:
```
git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
cd ../
```
Here we install via the <a href="https://github.com/pytorch-labs/attention-gym">attention-gym github</a> as it will ensure compatibility and give us access to their visualization tool. 


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
        description: a torch module that implements multiheaded self-attention via
        flex_attention.
        args:
            d_in: int, the dimension of the input tensor.
            d_out: int, the dimension of the output tensor.
            n_heads: int, the number of heads to use for the multiheaded
            self-attention.
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
        qkv = self.in_proj(x)

        # Split qkv and divide d_in into heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Permute shape of qkv for flex_attention
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Get queries, keys, values
        queries, keys, values = qkv 

        # Calculate attention via flex_attention
        attn = flex_attention(queries, keys, values, block_mask=block_mask)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        # Pass attention output through output projection
        attn = self.out_proj(attn)

        return attn, queries, keys
```

### mask_mod Functions:

### Padding Mask

```python
# Edit data to have random zero-padding
pad = torch.zeros(1, d_in).to(device)
pad_idxs = [(b, range(torch.randint(seq_len//2, seq_len + 1, (1,)).item(), seq_len)) for b in range(batch_size)]
for b, idxs in pad_idxs:
    input_data[b, idxs] = pad
```

We generate some fake data and randomly edit in zero-pad tokens to the sequences. 

```python
# Padding boolean mask
collapsed_input = input_data[:, :, 0]
collapsed_pad = pad[:, 0]
pads = torch.eq(collapsed_input, collapsed_pad).to(device)
```

We create a boolean padding mask (`pads`) from the input_data with its embedding dimension collapsed.

```python
# Create causal padding mask
causal_mask = causal
padding_mask = create_padding_mask(pads)
masks = [causal, padding_mask]
combined_mask = and_masks(*masks)
causal_padding_mask = create_block_mask(combined_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
```

Combines the `causal` and `padding` mask_mod functions using the torch `and_masks` function, allowing us to create a BlockMask.

```python
# Forward pass
attn_output, query, key = mhfa(input_data, causal_padding_mask)
```

Performs multiheaded attention to the `input_data`, applying our compiled custom attention mask.  

```python
# Visualize first sequence attention scores with mask
visualize_attention_scores(
    query,
    key,
    mask_mod=combined_mask,
    device=device,
    name="causal_padding_mask",
    path=Path("./causal_padding_mask.png"),
)
```

![visualized attention after masking](causal_padding_mask.png)

Visualizes the masked attention score of the first padded sequence.

## Resources used:
- https://pytorch.org/blog/flexattention/ 
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/
02_bonus_efficient-multihead-attention 
- https://github.com/ViktorooReps/llm-experiments/blob/59fe19a6fe6be2cd3652f75afcc90156953889cc/src/models/modelling_llama_long_context.py#L226
- https://github.com/pytorch-labs/attention-gym/issues/38