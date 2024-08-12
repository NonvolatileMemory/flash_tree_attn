

## Flash-Tree-Attention

The `Flash-Tree-Attention` library introduces an efficient approach to integrating speculative tree decoding with traditional flash decoding, optimizing the attention mechanism used in transformer models. This method enables parallel processing on the dimension of queries (`q`) against key-value (`k-v`) cache, facilitating rapid and dynamic attention calculations that support speculative tree decoding paths.

### Decoding Stage: `flash_decoding_attention`

This function is a critical component of the library. It handles the attention mechanism during the decoding stage.

#### Features

- **Parallel Query Processing**: Handles multiple queries in parallel by flash decoding with kv cache, enhancing the decoding efficiency.
- **Speculative Tree Decoding**: Integrates current keys and values to compute tree-based attention, allowing for speculative paths during decoding.

#### Function Signature

```python
def flash_decoding_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    mask: torch.Tensor,
    kv_seq_len: torch.Tensor,
    block_size: int = 64,
    max_seq_len_in_batch: int = None,
    output: torch.Tensor = None,
    mid_output: torch.Tensor = None,
    mid_output_lse: torch.Tensor = None,
    sm_scale: int = None,
    kv_group_num: int = 1,
):
```

#### Parameters

- **q**: The query tensor with shape `[bsz, num_heads, q_len, head_dim]`.
- **k_cache, v_cache**: Cached keys and values from previous sequences.
- **current_k, current_v**: Current keys and values for computing tree attention.
- **mask**: Mask tensor to avoid attention to unwanted positions.
- **kv_seq_len**: Tensor that records sequence lengths, incorporating past sequence lengths, not include current kv lenghts.
- **block_size**: The size of each block for splitting key-value pairs, default is 64.

#### Outputs

- **Output tensor**: The final output tensor after computing the attention, shaped `[bsz, num_heads, qlen, head_dim]`.

### Example Usage

```python
# Define tensors for q, k_cache, v_cache, etc.
# Call the function with appropriate parameters
output = flash_decoding_attention(q, k_cache, v_cache, current_k, current_v, mask, kv_seq_len)
```

### TODO

- [ ] replace the padding.
- [ ] benchmark.
- [ ] acceleration of short context tree attention.
- [ ] integration into transformers, medusa, and glide with a cape.
- [ ] kv cache management.

This library is designed for advanced users familiar with transformer architectures and speculative decoding techniques, providing a  tool for those looking to enhance the performance of their neural networks through optimized decoding strategies.

