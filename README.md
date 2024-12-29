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

### 


## Resources that helped me 
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention 
- https://pytorch.org/blog/flexattention/ 
- https://github.com/ViktorooReps/llm-experiments/blob/59fe19a6fe6be2cd3652f75afcc90156953889cc/src/models/modelling_llama_long_context.py#L226
- https://github.com/pytorch-labs/attention-gym/issues/38