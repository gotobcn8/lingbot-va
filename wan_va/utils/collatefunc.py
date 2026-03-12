import torch
import torch.nn.functional as F

def collate_get_mask(batch):
    # action shape [dim,Frames,]
    # batch['text_emb'] = torch.pad()
    text_embs = [b['text_emb'] for b in batch]
    latents = [b['latents'] for b in batch]
    text_emb_active_size = [t.shape[0] for t in text_embs]
    latents_active_size = [t.shape[1] for t in latents]
    max_tokens_size = max(text_emb_active_size)
    max_frames_size = max(latents_active_size)
    # max_actionchunk_size = max(t.shape[0] for t in batch['text_emb'])

    padded = []
    # def pad(embeds, max_tokens_size)
    for t in text_embs:
        pad_len = max_tokens_size - t.shape[0]
        t_pad = F.pad(t, (0, 0, 0, pad_len))  # (left,right,top,bottom)
        padded.append(t_pad)
    batch['text_emb'] = torch.stack(padded)


    padded = []
    for t in latents:
        pad_len = max_frames_size - t.shape[1]
        t_pad = F.pad(t, (
                0, 0,   # w
                0, 0,   # h
                0, pad_len
            ))  # frames
        padded.append(t_pad)
    batch['latents'] = torch.stack(padded)
    
    batch['text_emb_active_tokens'] = torch.tensor(text_emb_active_size)
    batch['latents_active_tokens'] = torch.tensor(latents_active_size)
    return batch
