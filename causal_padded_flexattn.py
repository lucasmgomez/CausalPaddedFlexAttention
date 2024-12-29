import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask


# For visualizing attention scores after masking
import sys
sys.path.append("./attention-gym")
from attn_gym import visualize_attention_scores
from pathlib import Path

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

def create_padding_mask(pads):
    def padding(b, h, q_idx, kv_idx):
        return ~pads[b, q_idx] & ~pads[b, kv_idx]
    return padding

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

if __name__ == "__main__":
    # Set seed and check torch version
    torch.manual_seed(28122024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}") # Make sure it's >= 2.5.1

    # MHFA paramters
    d_in = 64
    d_out = 64
    n_heads = 8

    # Create MHFA module
    mhfa = MultiheadFlexAttention(d_in, d_out, n_heads).to(device)

    # Data parameters
    batch_size = 1
    seq_len = 10

    # Create random inpuy data
    input_data = torch.randn(batch_size, seq_len, d_in).to(device)

    # Edit data to have random zero-padding
    pad = torch.zeros(1, d_in).to(device)
    pad_idxs = [(b, range(torch.randint(seq_len//2, seq_len + 1, (1,)).item(), seq_len)) for b in range(batch_size)]
    for b, idxs in pad_idxs:
        input_data[b, idxs] = pad

    # Padding boolean mask
    collapsed_input = input_data[:, :, 0]
    collapsed_pad = pad[:, 0]
    pads = torch.eq(collapsed_input, collapsed_pad).to(device)

    # Create causal padding mask
    causal_mask = causal
    padding_mask = create_padding_mask(pads)
    masks = [causal, padding_mask]
    combined_mask = and_masks(*masks)
    causal_padding_mask = create_block_mask(combined_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)

    # Forward pass
    attn_output, query, key = mhfa(input_data, causal_padding_mask)

    # Print where padding is
    print("Padding of first sequence is at: ", list(pad_idxs[0][1]))

    # Visualize first sequence attention scores with mask
    visualize_attention_scores(
        query,
        key,
        mask_mod=combined_mask,
        device=device,
        name="causal_padding_mask",
        path=Path("./causal_padding_mask.png"),
    )