from __future__ import annotations

from hif4_system.formats import dequantize_nvfp4, standard_hif4_quantize


def _standard(quant, scale):
    return standard_hif4_quantize(dequantize_nvfp4(quant, scale))


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    return {
        "weight_params": _standard(weight_quant, weight_scale),
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(activation_quant, activation_scale, activation_state):
    return _standard(activation_quant, activation_scale)


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _standard(q_quant, q_scale)


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _standard(k_quant, k_scale)


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _standard(v_quant, v_scale)
