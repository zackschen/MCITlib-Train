from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _flash_attn_varlen_qkvpacked_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func as _flash_attn_varlen_qkvpacked_func
    except ImportError:
        _flash_attn_varlen_qkvpacked_func = None

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as _flash_attn_qkvpacked_func
except ImportError:
    _flash_attn_qkvpacked_func = None

try:
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    try:
        from flash_attn.padding import unpad_input, pad_input
    except ImportError:
        def unpad_input(hidden_states, attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            return hidden_states[indices], indices, cu_seqlens, max_seqlen_in_batch

        def pad_input(hidden_states, indices, batch, seqlen):
            output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
            output[indices] = hidden_states
            return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def _torch_qkv_attention(qkv, attention_mask=None):
    q = qkv[:, :, 0].transpose(1, 2)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)

    if attention_mask is None:
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
    else:
        seq_len = qkv.shape[1]
        causal_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=qkv.device
        ).tril()
        attn_mask = causal_mask[None, None, :, :] & attention_mask[:, None, None, :].bool()
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    return output.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], -1)


def _flash_qkv_attention(qkv, attention_mask=None):
    if attention_mask is None and _flash_attn_qkvpacked_func is not None:
        return _flash_attn_qkvpacked_func(
            qkv, 0.0, softmax_scale=None, causal=True
        ).reshape(qkv.shape[0], qkv.shape[1], -1)

    if _flash_attn_varlen_qkvpacked_func is None:
        return _torch_qkv_attention(qkv, attention_mask)

    batch_size, seq_len = qkv.shape[:2]
    if attention_mask is None:
        qkv = qkv.reshape(-1, *qkv.shape[2:])
        cu_q_lens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        output = _flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, seq_len, 0.0, softmax_scale=None, causal=True
        )
        return output.reshape(batch_size, seq_len, -1)

    unpadded_qkv, indices, cu_q_lens, max_s = unpad_input(
        qkv.reshape(batch_size, seq_len, -1), attention_mask
    )
    unpadded_qkv = unpadded_qkv.view(-1, *qkv.shape[2:])
    output_unpad = _flash_attn_varlen_qkvpacked_func(
        unpadded_qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output_unpad = output_unpad.reshape(-1, qkv.shape[3] * qkv.shape[4])
    return pad_input(output_unpad, indices, batch_size, seq_len)


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    output = _flash_qkv_attention(qkv, key_padding_mask)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
