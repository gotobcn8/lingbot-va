import math

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
    Future frame alignment
    
    Args:
        a: (B, Fa, T, D) - source repr
        b: (B, Fb, T, D) - target repr ()
        K: aligned future frames
        temperature: temperature coef
        mask_type: 'triangular' (low triangle) or 'window' (slide window)
        
    Returns:
        loss: scalar loss
    """
    # B, Fa, T, D = a.shape
    # _, Fb, _, _ = b.shape
    B, Fa, D = a.shape
    _, Fb, _ = b.shape
    # print(B,Fa,D,Fb)

    a = a.reshape(-1, Tokens, D)
    b = b.reshape(-1, Tokens, D)
    Fa, Fb = a.shape[0], b.shape[0]
    # 1. normalization feature
    a_norm = F.normalize(a, dim=-1)  # (B, Fa, T, D)
    b_norm = F.normalize(b, dim=-1)  # (B, Fb, T, D)
    
    # 2. compute token-level similarity matrix
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
            for j in range(i, min(i+K, Fb)):
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

def motion_incremental_alignment(
    a,
    b,
    Tokens=192,
    pool="mean",
    normalize_before_delta=False,
    normalize_after_delta=True,
    eps=1e-8,
):
    """
    Motion incremental alignment via pooled temporal delta.

    Args:
        a: Tensor, shape (B, F*Tokens, D)
        b: Tensor, shape (B, F*Tokens, D)
        Tokens: int, number of tokens per frame/chunk
        pool: "mean" or "max"
        normalize_before_delta: whether to normalize pooled frame features before differencing
        normalize_after_delta: whether to normalize delta features before cosine loss
        eps: numerical stability

    Returns:
        loss: scalar
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} vs {b.shape}")

    if a.dim() != 3:
        raise ValueError(f"Expected a and b to be 3D tensors of shape (B, F*Tokens, D), got {a.shape}")

    B, L, D = a.shape

    if L % Tokens != 0:
        raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")

    F_steps = L // Tokens

    # (B, F, Tokens, D)
    a = a.view(B, F_steps, Tokens, D)
    b = b.view(B, F_steps, Tokens, D)

    # Pool token dimension -> (B, F, D)
    if pool == "mean":
        a = a.mean(dim=2)
        b = b.mean(dim=2)
    elif pool == "max":
        a = a.max(dim=2).values
        b = b.max(dim=2).values
    else:
        raise ValueError(f"Unknown pool type: {pool}")

    if normalize_before_delta:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)

    # Temporal delta: (B, F-1, D)
    delta_a = a[:, 1:] - a[:, :-1]
    delta_b = b[:, 1:] - b[:, :-1]

    if delta_a.shape[1] == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    if normalize_after_delta:
        delta_a = F.normalize(delta_a, dim=-1, eps=eps)
        delta_b = F.normalize(delta_b, dim=-1, eps=eps)

    # Cosine similarity over delta features
    cos_sim = (delta_a * delta_b).sum(dim=-1)   # (B, F-1)
    loss = 1.0 - cos_sim.mean()

    return loss

def motion_incremental_alignment_tokenwise(a, b, Tokens=192, eps=1e-8):
    """
    Token-wise motion incremental alignment.

    Args:
        a, b: (B, F*Tokens, D). b is treated as the fixed target.

    Returns:
        loss: scalar
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have same shape, got {a.shape} vs {b.shape}")

    B, L, D = a.shape
    if L % Tokens != 0:
        raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")

    F_steps = L // Tokens

    # (B, F, Tokens, D)
    a = a.view(B, F_steps, Tokens, D)
    b = b.detach().view(B, F_steps, Tokens, D)

    # temporal delta per token
    delta_a = a[:, 1:] - a[:, :-1]   # (B, F-1, Tokens, D)
    delta_b = b[:, 1:] - b[:, :-1]

    if delta_a.shape[1] == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    delta_a = F.normalize(delta_a, dim=-1, eps=eps)
    delta_b = F.normalize(delta_b, dim=-1, eps=eps)

    cos_sim = (delta_a * delta_b).sum(dim=-1)   # (B, F-1, Tokens)
    loss = 1.0 - cos_sim.mean()

    return loss


def build_topk_dest_prob(
    dest_score,
    topk_ratio=0.1,
    tau=3.0,
    eps=1e-6,
):
    """
    Build a destination-token distribution over the top-scoring tokens.

    Args:
        dest_score: [B, T]
    """
    if dest_score.dim() != 2:
        raise ValueError(f"Expected dest_score with shape [B, T], got {dest_score.shape}")

    _, tokens = dest_score.shape
    k = max(1, min(tokens, int(math.ceil(tokens * topk_ratio))))
    topk_idx = torch.topk(dest_score, k, dim=-1).indices

    masked_score = dest_score.new_full(dest_score.shape, -torch.inf)
    masked_score.scatter_(dim=-1, index=topk_idx, src=dest_score.gather(-1, topk_idx))
    prob = torch.softmax(masked_score / max(tau, eps), dim=-1)
    return prob / (prob.sum(dim=-1, keepdim=True) + eps)


def destination_loss_safe(
    h,
    ta,
    Tokens=192,
    tau_dest=3.0,
    tau_sim=0.1,
    tau_src=3.0,
    topk_ratio=0.1,
    eps=1e-6,
):
    """
    Destination alignment loss.

    Args:
        h:  [B, F, T, D] or flattened [B, F*T, D]
        ta: [B, F, T, D] or flattened [B, F*T, D]
    """
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {h.shape} vs {ta.shape}")

    if h.dim() == 3:
        B, L, D = h.shape
        if L % Tokens != 0:
            raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")
        h = h.reshape(B, L // Tokens, Tokens, D)
        ta = ta.reshape(B, L // Tokens, Tokens, D)
    elif h.dim() != 4:
        raise ValueError(f"Expected h and ta with shape [B,F,T,D] or [B,F*T,D], got {h.shape}")

    if h.shape[1] == 0:
        return h.new_zeros(())

    h_final = h[:, -1]
    ta_final = ta[:, -1].detach()

    h_final = F.normalize(h_final.float(), dim=-1, eps=eps)
    ta_final = F.normalize(ta_final.float(), dim=-1, eps=eps)

    sim = torch.matmul(h_final, ta_final.transpose(-1, -2))
    log_pred = F.log_softmax(sim / max(tau_sim, eps), dim=-1)

    dest_score = ta[:, -1].detach().float().norm(dim=-1)
    dest_prob = build_topk_dest_prob(
        dest_score,
        topk_ratio=topk_ratio,
        tau=tau_dest,
        eps=eps,
    ).detach()

    target = dest_prob.unsqueeze(1).expand_as(log_pred)

    src_score = (ta[:, -1].detach().float() - ta[:, 0].detach().float()).norm(dim=-1)
    src_weight = F.softmax(src_score / max(tau_src, eps), dim=-1).detach()

    kl_each = F.kl_div(
        log_pred,
        target,
        reduction="none",
    ).sum(dim=-1)

    loss = (src_weight * kl_each).sum() / (src_weight.sum() + eps)
    return loss


def unified_dest_and_motion(
    h,
    ta,
    Tokens=192,
    tau_dest=3.0,
    tau_sim=0.1,
    tau_src=3.0,
    topk_ratio=0.1,
    eps=1e-6,
):
    loss_dest = destination_loss_safe(
        h,
        ta,
        Tokens=Tokens,
        tau_dest=tau_dest,
        tau_sim=tau_sim,
        tau_src=tau_src,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    
    loss_motion = motion_incremental_alignment_tokenwise(h, ta, Tokens=Tokens, eps=eps)
    
    return {
        'dest_loss': loss_dest,
        'motion_loss': loss_motion,
    }
