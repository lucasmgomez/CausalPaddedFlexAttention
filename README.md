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
Here we install via the <a href="https://github.com/pytorch-labs/attention-gym">attention-gym git-hub</a> as it will ensure compatibility and give us access to their visualization tool. 


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

In order to actually use flex_attention in a transformer-like model we will need to implement it in a multi-headed attention module.

```python
class MultiheadFlexAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, bias=False):
        """
        description: a torch module that implements multi-headed
        self-attention via flex_attention.
        args:
            d_in: int, the dimension of the input tensor.
            d_out: int, the dimension of the output tensor.
            n_heads: int, the number of heads to use for the multi-headed
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
```
We define the necessary parameters like input and output dimensionalities, as well as the linear projections. 


```python
    def forward(self, x, block_mask):
        """
        description: forward pass of the multi-headed self-attention module.
        args:
            x: torch.Tensor, the input tensor of shape (batch_size, max_seq_len, d_in)
            block_mask: torch.Tensor, the block mask to use for flex_attention
        """
        batch_size, max_seq_len, d_in = x.shape

        # Create stacked qkv via input projection
        qkv = self.in_proj(x)

        # Split qkv and divide d_in into heads
        qkv = qkv.view(batch_size, max_seq_len, 3, self.n_heads, self.d_head)

        # Permute shape of qkv for flex_attention
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Get queries, keys, values
        queries, keys, values = qkv 

        # Calculate attention via flex_attention
        attn = flex_attention(queries, keys, values, block_mask=block_mask)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.d_out)

        # Pass attention output through output projection
        attn = self.out_proj(attn)

        return attn, queries, keys
```

Here we use a forward function very similar to that of the [MultiheadAttention](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention) class from PyTorch. The main difference being that we pass in a block_mask (more on that soon) and use the flex_attention function to perform the self-attention across heads. 

### mask_mod Functions:

The key feature of FlexAttention is the ability to develop and use custom attention masks efficiently without having to write custom kernels.

To use this feature we need to first define are masks as boolean tensors. First lets do a causal mask as its simple and provided by the FlexAttention developers in their blog.

#### Causal Mask

```python
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```

Here `b` and `h` refer to the batch size and number of attention heads respectively. With `q_idx`  and `kv_idx` referring to postions in the query and key/value respectively. For example if your input data has a sequence length of `5` then `q_idx = torch.Tensor([0,1,2,3,4])`.

This function returns a causal boolean attention mask because attention scores will only be calculated with key/value pairs at or before the query token. 

Okay now lets handle padded inputs with a padding mask. 

#### Padding Mask

Unlike our causal mask our padding mask is batch dependent, as its boolean values depend on the sequence specific padding. To handle this we need to use a padding lookup to determine which tokens of a given sequence are padding and should be ignored. 

```python
def create_padding_mask(pads):
    def padding(b, h, q_idx, kv_idx):
        return ~pads[b, q_idx] & ~pads[b, kv_idx]
    return padding
```

Here `pads` is a BoolTensor of size `(batch_size, max_seq_len)` where positions of padding tokens are True and the other token positions are False. The `padding` mask_mod function returns a padding attention mask because attention scores will only be calculated when neither the query nor key/value token are padding. 

### Dummy Data

Before putting these masks together and using them in our MultiheadFlexAttention class we need to define some parameters and create dummy data.

```python
# MHFA paramters
d_in = 64
d_out = 64
n_heads = 8

# Create MHFA module
mhfa = MultiheadFlexAttention(d_in, d_out, n_heads).to(device)

# Data parameters
batch_size = 1 # any batch_size works
max_seq_len = 10

# Create random input data
input_data = torch.randn(batch_size, max_seq_len, d_in).to(device)
```

Lets edit the `input_data` to have some random end of sequence zero-padding.

```python
# Edit data to have random zero-padding
pad = torch.zeros(1, d_in).to(device)
pad_idxs = [(b, range(torch.randint(max_seq_len//2, max_seq_len + 1, (1,)).item(), max_seq_len)) for b in range(batch_size)]
for b, idxs in pad_idxs:
    input_data[b, idxs] = pad
```

Now that we have our padded input data we needs to create the padding lookup table for our `padding` mask_mod function. 

```python
# Padding boolean mask
collapsed_input = input_data[:, :, 0] # (batch_size, max_seq_len)
collapsed_pad = pad[:, 0] # (1,)
pads = torch.eq(collapsed_input, collapsed_pad).to(device)
```

The mask_mod functions don't take into account the `input_data`'s embedding dimension, so we need to make the padding lookup (`pads`) with that dimension collapsed. 

### Causal + Padding BlockMasks

Now we have everything we need to create our causal-padding attention mask

```python
# Create causal padding mask
causal_mask = causal
padding_mask = create_padding_mask(pads)
masks = [causal, padding_mask]
combined_mask = and_masks(*masks)
causal_padding_mask = create_block_mask(combined_mask, B=batch_size, H=None, Q_LEN=max_seq_len, KV_LEN=max_seq_len, _compile=True)
```

Here we combine the `causal` and `padding` mask_mod functions using the torch `and_masks` function, allowing us to create a single BlockMask. 

_Note: the developers suggest toggling the `_compile` argument makes creating BlockMasks much faster, which is crucial when they are batch-dependent._

```python
# Forward pass
attn_output, query, key = mhfa(input_data, causal_padding_mask)
```

Now we can use our MultiheadFlexAttention class to perform self-attention on the `input_data`, applying our compiled custom attention mask.  

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

As you can see the attention scores for both padding and future tokens are properly masked. 

## Resources used:
- https://pytorch.org/blog/flexattention/ 
- [https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention)
- https://github.com/ViktorooReps/llm-experiments/blob/59fe19a6fe6be2cd3652f75afcc90156953889cc/src/models/modelling_llama_long_context.py#L226
- https://github.com/pytorch-labs/attention-gym/issues/38