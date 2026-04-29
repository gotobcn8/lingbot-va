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
    Future frame prediction alignment.
    
    Args:
        a: (B, F*Tokens, D) - predicted future representation from model-side MLP
        b: (B, F*Tokens, D) - target trace representation
        K: future offset, aligning a[t] with b[t + K]
        
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

    valid_steps = min(Fa, Fb - K)
    if valid_steps <= 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    pred = a[:, :valid_steps]
    target = b[:, K:K + valid_steps]

    mse_loss = F.mse_loss(pred.float(), target.float())
    cos_sim = F.cosine_similarity(pred.float(), target.float(), dim=-1, eps=eps)
    loss = mse_loss + 0.1 * (1.0 - cos_sim.mean())

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

def motion_incremental_alignment_tokenwise(a, b, K = 1, Tokens=192, eps=1e-8, **_):
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
    delta_a = a[:, K:] - a[:, :-K]   # (B, F-1, Tokens, D)
    delta_b = b[:, K:] - b[:, :-K]

    if delta_a.shape[1] == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    loss = F.mse_loss(delta_a.float(), delta_b.float())

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
