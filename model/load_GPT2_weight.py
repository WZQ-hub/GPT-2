import torch
import numpy as np

# --- 1. 配置信息 (作为常量保留) ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


# --- 2. 核心功能函数 (纯工具，import 时不触发运行) ---

def assign(left, right):
    """辅助函数：检查形状并转换张量"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.from_numpy(right) if isinstance(right, np.ndarray) else torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    专门负责将下载好的参数字典 params 加载进 GPT2 模型对象 gpt 中
    """
    # 加载 Token 和 Position Embeddings
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])

    # 循环加载每一个 Transformer Block
    for b in range(len(params["blocks"])):
        block_params = params["blocks"][b]
        target_block = gpt.trf_blocks[b]

        # A. Attention 部分 (处理 c_attn 拆分)
        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)

        target_block.att.W_query.weight = assign(target_block.att.W_query.weight, q_w.T)
        target_block.att.W_query.bias = assign(target_block.att.W_query.bias, q_b)
        target_block.att.W_key.weight = assign(target_block.att.W_key.weight, k_w.T)
        target_block.att.W_key.bias = assign(target_block.att.W_key.bias, k_b)
        target_block.att.W_value.weight = assign(target_block.att.W_value.weight, v_w.T)
        target_block.att.W_value.bias = assign(target_block.att.W_value.bias, v_b)

        # B. Attention 输出投影
        target_block.att.out_proj.weight = assign(target_block.att.out_proj.weight,
                                                  block_params["attn"]["c_proj"]["w"].T)
        target_block.att.out_proj.bias = assign(target_block.att.out_proj.bias, block_params["attn"]["c_proj"]["b"])

        # C. FeedForward (MLP) 部分
        target_block.ff.layers[0].weight = assign(target_block.ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T)
        target_block.ff.layers[0].bias = assign(target_block.ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"])
        target_block.ff.layers[2].weight = assign(target_block.ff.layers[2].weight,
                                                  block_params["mlp"]["c_proj"]["w"].T)
        target_block.ff.layers[2].bias = assign(target_block.ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"])

        # D. LayerNorm 部分
        target_block.norm1.scale = assign(target_block.norm1.scale, block_params["ln_1"]["g"])
        target_block.norm1.shift = assign(target_block.norm1.shift, block_params["ln_1"]["b"])
        target_block.norm2.scale = assign(target_block.norm2.scale, block_params["ln_2"]["g"])
        target_block.norm2.shift = assign(target_block.norm2.shift, block_params["ln_2"]["b"])

    # 加载最终的 LayerNorm 和输出层
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    # print(f"Successfully loaded weights into GPT model.")

