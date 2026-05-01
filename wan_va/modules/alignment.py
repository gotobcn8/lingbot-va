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

def future_alignment_loss(a, b, K=3, Tokens=192, temperature=1.0, mask_type='window', eps=1e-8):
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
    if a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]:
        raise ValueError(f"a and b must have matching batch and hidden dims, got {a.shape} vs {b.shape}")
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"Expected a and b to be 3D tensors of shape (B, F*Tokens, D), got {a.shape} vs {b.shape}")

    B, La, D = a.shape
    _, Lb, _ = b.shape
    if La % Tokens != 0 or Lb % Tokens != 0:
        raise ValueError(f"Sequence lengths {La} and {Lb} must be divisible by Tokens={Tokens}")

    Fa, Fb = La // Tokens, Lb // Tokens
    a = a.reshape(B, Fa, Tokens, D)
    b = b.reshape(B, Fb, Tokens, D)

    a_norm = F.normalize(a, dim=-1, eps=eps)
    b_norm = F.normalize(b, dim=-1, eps=eps)
    cos_sim = torch.einsum('bftd,bgtd->bftg', a_norm, b_norm)
    
    # 3. 创建对齐 mask
    if mask_type == 'triangular':
        # a[t] 对齐 b[t+1:] (需要 Fa == Fb)
        if Fa != Fb:
            raise ValueError(f"Triangular mask requires Fa == Fb, but got Fa={Fa}, Fb={Fb}")
        frame_mask = torch.triu(torch.ones(Fa, Fb, dtype=torch.bool), diagonal=1)
        
    elif mask_type == 'window':
        # a[t] 对齐 b[t+1:t+K+1] (支持 Fa != Fb)
        frame_mask = torch.zeros(Fa, Fb, dtype=torch.bool)
        for i in range(Fa):
            for j in range(i + 1, min(i + K + 1, Fb)):
                frame_mask[i, j] = True
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    frame_mask = frame_mask.to(a.device)  # (Fa, Fb)
    
    token_mask = frame_mask[None, :, None, :].expand(B, Fa, Tokens, Fb)
    valid_sims = cos_sim[token_mask]
    
    if valid_sims.numel() == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    
    # 5. 计算损失
    loss = 1 - valid_sims.mean()
    
    # 可选: 使用 temperature 缩放
    if temperature != 1.0:
        loss = 1 - (valid_sims / temperature).mean()

    # print(f"frame_mask has {frame_mask.sum().item()} True entries")
    return loss

def future_alignment_loss_excludeself(a, b, K=3, Tokens=192, temperature=1.0, mask_type='window'):
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
    return future_alignment_loss(
        a,
        b,
        K=K,
        Tokens=Tokens,
        temperature=temperature,
        mask_type=mask_type,
    )

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

def motion_incremental_alignment_tokenwise(a, b, Tokens=192, eps=1e-8, **_):
    """
    Token-wise motion incremental alignment.

    Args:
        a, b: (B, F*Tokens, D)

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
    b = b.view(B, F_steps, Tokens, D)

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


def UnifiedTraceAlign(
    a,
    b,
    K=3,
    Tokens=192,
    temperature=1.0,
    mask_type='window',
    future_weight=1.0,
    motion_weight=1.0,
    eps=1e-8,
    # return_components=False,
):
    """
    A Unified Token-wise motion incremental and future planning alignment.

    Args:
        a, b: (B, F*Tokens, D)
        K: number of future frames for the window mask
        future_weight: weight for future frame alignment
        motion_weight: weight for token-wise temporal delta alignment
        return_components: return (total_loss, future_loss, motion_dynamic_loss)

    Returns:
        loss: scalar
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have same shape, got {a.shape} vs {b.shape}")

    if a.dim() != 3:
        raise ValueError(f"Expected a and b to be 3D tensors of shape (B, F*Tokens, D), got {a.shape}")

    B, L, D = a.shape
    if L % Tokens != 0:
        raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")

    F_steps = L // Tokens

    # (B, F, Tokens, D)
    a = a.reshape(B, F_steps, Tokens, D)
    b = b.reshape(B, F_steps, Tokens, D)

    # -------------------------------------Future Loss-------------------------------------
    if mask_type == 'triangular':
        frame_mask = torch.triu(
            torch.ones(F_steps, F_steps, dtype=torch.bool, device=a.device),
            diagonal=1,
        )
    elif mask_type == 'window':
        frame_mask = torch.zeros(F_steps, F_steps, dtype=torch.bool, device=a.device)
        for i in range(F_steps):
            for j in range(i + 1, min(i + K + 1, F_steps)):
                frame_mask[i, j] = True
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    if frame_mask.any():
        a_norm = F.normalize(a, dim=-1, eps=eps)
        b_norm = F.normalize(b, dim=-1, eps=eps)
        cos_sim = torch.einsum('bftd,bgtd->bftg', a_norm, b_norm)

        token_mask = frame_mask[None, :, None, :].expand(B, F_steps, Tokens, F_steps)
        valid_sims = cos_sim[token_mask]

        future_loss = 1.0 - valid_sims.mean()
        if temperature != 1.0:
            future_loss = 1.0 - (valid_sims / temperature).mean()
    else:
        future_loss = torch.zeros((), device=a.device, dtype=a.dtype)
    
    # -------------------------------------Motion Dynamic Loss-------------------------------------
    delta_a = a[:, 1:] - a[:, :-1]   # (B, F-1, Tokens, D)
    delta_b = b[:, 1:] - b[:, :-1]

    if delta_a.shape[1] == 0:
        motion_dynamic_loss = torch.zeros((), device=a.device, dtype=a.dtype)
    else:
        delta_a = F.normalize(delta_a, dim=-1, eps=eps)
        delta_b = F.normalize(delta_b, dim=-1, eps=eps)

        cos_sim = (delta_a * delta_b).sum(dim=-1)   # (B, F-1, Tokens)
        motion_dynamic_loss = 1.0 - cos_sim.mean()

    # future_loss *= future_weight 
    # motion_dynamic_loss *= motion_weight

    # if return_components:
    return future_loss, motion_dynamic_loss

    # return loss


def build_ta_feature_teacher(
    ta_feat,
    top_ratio=0.2,
    use_delta=True,
    temperature=None,
    eps=1e-6,
):
    """
    ta_feat:
        [B, F, N, D]

    return:
        ta_teacher [B, F, N]
        每一帧内归一化，只表示 TA 认为重要的 token。
    """

    B, F_len, N, D = ta_feat.shape

    if use_delta:
        # temporal feature change
        delta = ta_feat[:, 1:] - ta_feat[:, :-1]      # [B, F-1, N, D]
        score = delta.norm(dim=-1)                   # [B, F-1, N]

        # pad first frame, so shape becomes [B, F, N]
        first = score[:, :1]
        score = torch.cat([first, score], dim=1)      # [B, F, N]
    else:
        # feature magnitude
        score = ta_feat.norm(dim=-1)                 # [B, F, N]

    # optional softmax teacher
    if temperature is not None:
        ta_teacher = torch.softmax(score / temperature, dim=-1)
        return ta_teacher

    # top-k teacher, more robust
    k = max(1, int(top_ratio * N))
    threshold = torch.topk(score, k, dim=-1).values[..., -1:]  # [B, F, 1]

    ta_teacher = (score >= threshold).float()         # [B, F, N]

    # normalize only selected tokens
    ta_teacher = ta_teacher / (ta_teacher.sum(dim=-1, keepdim=True) + eps)

    return ta_teacher


def ta_to_attention_teacher(ta_feat, use_delta=False):
    """
    Convert TraceAnything features to per-token teacher logits.

    Args:
        ta_feat: Tensor of shape [B, F, N, D].
        use_delta: When True, use temporal feature change as the token score.

    Returns:
        score: Tensor of shape [B, F, N]. Apply softmax over the last dim to
            get the teacher distribution.
    """
    if ta_feat.dim() != 4:
        raise ValueError(f"Expected ta_feat with shape [B, F, N, D], got {ta_feat.shape}")

    if use_delta:
        delta = ta_feat[:, 1:] - ta_feat[:, :-1]
        score = delta.norm(dim=-1)
        if score.shape[1] == 0:
            score = ta_feat.new_zeros(ta_feat.shape[:3])
        else:
            score = torch.cat([score[:, :1], score], dim=1)
    else:
        score = ta_feat.norm(dim=-1)

    return score


def extract_key_attention(attn, frames=None, tokens=None):
    """
    Aggregate attention weights into key-token importance [B, F, N].

    Supported inputs:
        [B, F, N]: already frame-token key attention.
        [B, H, F, Q, N]: heads and query tokens are averaged.
        [B, H, Q, K]: heads and queries are averaged, then K is reshaped by
            frames/tokens or inferred from a square self-attention map.
        [B, H, Fq, Nq, Fk, Nk]: heads and all query axes are averaged.
    """
    if attn.dim() == 3:
        return attn

    if attn.dim() == 5:
        # [B, H, F, Q, N] -> [B, F, N]
        return attn.mean(dim=(1, 3))

    if attn.dim() == 6:
        # [B, H, Fq, Nq, Fk, Nk] -> [B, Fk, Nk]
        return attn.mean(dim=(1, 2, 3))

    if attn.dim() != 4:
        raise ValueError(
            "Expected attn with shape [B,F,N], [B,H,F,Q,N], [B,H,Q,K], "
            f"or [B,H,Fq,Nq,Fk,Nk], got {attn.shape}"
        )

    B, _, _, key_len = attn.shape
    key_attention = attn.mean(dim=(1, 2))

    if frames is None and tokens is None:
        square = int(key_len ** 0.5)
        if square * square != key_len:
            raise ValueError(
                "frames or tokens must be provided when key length is not a square, "
                f"got key_len={key_len}"
            )
        frames, tokens = square, square
    elif frames is None:
        if key_len % tokens != 0:
            raise ValueError(f"key_len={key_len} is not divisible by tokens={tokens}")
        frames = key_len // tokens
    elif tokens is None:
        if key_len % frames != 0:
            raise ValueError(f"key_len={key_len} is not divisible by frames={frames}")
        tokens = key_len // frames

    if frames * tokens != key_len:
        raise ValueError(
            f"frames * tokens must equal key_len, got {frames} * {tokens} != {key_len}"
        )

    return key_attention.reshape(B, frames, tokens)


def ta_attention_reward_loss(
    attn,
    ta_feat=None,
    ta_teacher=None,
    frames=None,
    tokens=None,
    teacher_temperature=1.0,
    teacher_use_delta=True,
    eps=1e-6,
):
    """
    Training-only reward loss for matching model attention to TA token saliency.

    Implements:
        ta_teacher = softmax(ta_to_attention_teacher(ta_feat), dim=-1)
        attn_key = extract_key_attention(attn)
        attn_key = attn_key / (attn_key.sum(dim=-1, keepdim=True) + eps)
        reward = (attn_key * ta_teacher.detach()).sum(dim=-1).mean()
        loss_attn = -reward
    """
    if ta_teacher is None:
        if ta_feat is None:
            raise ValueError("Either ta_feat or ta_teacher must be provided")
        ta_teacher = ta_to_attention_teacher(ta_feat, use_delta=teacher_use_delta)
        ta_teacher = torch.softmax(ta_teacher / teacher_temperature, dim=-1)
    else:
        ta_teacher = ta_teacher.to(device=attn.device)

    attn_key = extract_key_attention(attn, frames=frames, tokens=tokens)
    if attn_key.shape != ta_teacher.shape:
        raise ValueError(
            f"attn_key and ta_teacher must have the same shape, got "
            f"{attn_key.shape} vs {ta_teacher.shape}"
        )

    attn_key = attn_key / (attn_key.sum(dim=-1, keepdim=True) + eps)
    reward = (attn_key * ta_teacher.detach()).sum(dim=-1).mean()

    return -reward
