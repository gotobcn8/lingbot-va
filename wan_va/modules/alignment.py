import torch
import torch.nn.functional as F

def create_future_alignment_mask(F, K, device='cuda'):
    """
    创建未来帧对齐的 mask
    
    Args:
        F: Total F frames
        K: future K frames (a[t] 对齐 b[t+1] 到 b[t+K])
        
    Returns:
        mask: (F, F) bool mask, True represent the loss computation part.
    """
    mask = torch.zeros(F, F, dtype=torch.bool, device=device)
    
    for t in range(F):
        # a[t] align from b[t+1] to b[t+K]
        future_frames = range(t+1, min(t+K+1, F))
        for future_t in future_frames:
            mask[t, future_t] = True
            
    return mask


def future_alignment_loss(a, b, K=3, Tokens = 192, temperature=1.0, mask_type='window'):
    """
    计算未来帧对齐损失
    
    Args:
        a: (B, Fa, T, D) - 源表征（当前帧）
        b: (B, Fb, T, D) - 目标表征（未来帧）
        K: 对齐的未来帧数
        temperature: 温度系数
        mask_type: 'triangular' (下三角) 或 'window' (滑动窗口)
        
    Returns:
        loss: 标量损失
    """
    # B, Fa, T, D = a.shape
    # _, Fb, _, _ = b.shape
    B, Fa, D = a.shape
    _, Fb, _ = b.shape
    # print(B,Fa,D,Fb)

    a = a.reshape(-1, Tokens, D)
    b = b.reshape(-1, Tokens, D)
    Fa, Fb = a.shape[0], b.shape[0]
    # 1. 归一化特征
    a_norm = F.normalize(a, dim=-1)  # (B, Fa, T, D)
    b_norm = F.normalize(b, dim=-1)  # (B, Fb, T, D)
    
    # 2. 计算 token-level 相似度矩阵
    # 对每个 batch 独立计算，并考虑 token 对应关系
    # 形状: (B, Fa, T, Fb, T) - 太大会爆内存，改用更高效的方式
    # 由于 B 始终为 1，可以简化
    if B == 1:
        # 去掉 batch 维度
        a_norm = a_norm.squeeze(0)  # (Fa, T, D)
        b_norm = b_norm.squeeze(0)  # (Fb, T, D)
        
        # 计算相似度: 相同 token 位置之间计算
        cos_sim = torch.einsum('ftd,gtd->ftg', a_norm, b_norm)  # (Fa, T, Fb)
    else:
        # 如果 B > 1，使用批量计算
        # 先 reshape: (B, Fa*T, D) 和 (B, Fb*T, D)
        a_flat = a_norm.view(B, Fa * T, D)
        b_flat = b_norm.view(B, Fb * T, D)
        
        # 计算相似度矩阵: (B, Fa*T, Fb*T)
        cos_sim_flat = torch.bmm(a_flat, b_flat.transpose(1, 2))
        
        # 然后 reshape 回 (B, Fa, T, Fb, T)
        cos_sim = cos_sim_flat.view(B, Fa, T, Fb, T)
    
    # 3. 创建对齐 mask
    if mask_type == 'triangular':
        # a[t] 对齐 b[t+1:] (需要 Fa == Fb)
        if Fa != Fb:
            raise ValueError(f"Triangular mask requires Fa == Fb, but got Fa={Fa}, Fb={Fb}")
        frame_mask = torch.tril(torch.ones(Fa, Fb), diagonal=-1).bool()
        
    elif mask_type == 'window':
        # a[t] 对齐 b[t+1:t+K+1] (支持 Fa != Fb)
        frame_mask = torch.zeros(Fa, Fb, dtype=torch.bool)
        for i in range(Fa):
            for j in range(i+1, min(i+K+1, Fb)):
                frame_mask[i, j] = True
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    frame_mask = frame_mask.to(a.device)  # (Fa, Fb)
    
    # 4. 计算损失
    if B == 1:
        # 处理单个 batch
        # 扩展到 token 维度
        token_mask = frame_mask[:, None, :]  # (Fa, 1, Fb)
        token_mask = token_mask.expand(Fa, Tokens, Fb)  # (Fa, T, Fb)
        
        # 提取有效相似度
        valid_sims = cos_sim[token_mask]  # (num_pairs,)
    else:
        # 处理多个 batch
        # 扩展 mask 到 batch 和 token 维度
        token_mask = frame_mask[None, :, None, :]  # (1, Fa, 1, Fb)
        token_mask = token_mask.expand(B, Fa, Tokens, Fb)  # (B, Fa, T, Fb)
        
        # 提取有效相似度（需要先 reshape）
        cos_sim_reshaped = cos_sim.view(B, Fa, Tokens, Fb)  # 忽略最后一个 T 维度，因为我们只关心相同 token 位置
        valid_sims = cos_sim_reshaped[token_mask]  # (num_pairs,)
    
    if len(valid_sims) == 0:
        return torch.tensor(0.0, device=a.device)
    
    # 5. 计算损失
    loss = 1 - valid_sims.mean()
    
    # 可选: 使用 temperature 缩放
    if temperature != 1.0:
        loss = 1 - (valid_sims / temperature).mean()
    
    # print(f"frame_mask has {frame_mask.sum().item()} True entries")
    return loss