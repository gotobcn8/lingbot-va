# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import math
import os
import sys
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)

from modules.alignment import (
    motion_incremental_alignment,
    future_alignment_loss,
    motion_incremental_alignment_tokenwise,
    UnifiedTraceAlign,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler,
    collate_get_mask,
    modelswitch
)
from dataset import MultiLatentLeRobotDataset
import gc
# from remote_pdb import RemotePdb
import torch.multiprocessing as mp
import pdb
from datetime import datetime

FIRST = True

class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            keyword = getattr(config, 'keyword', '')
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name = f'{keyword}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        # print(config.max_tokens)
        
        ## Trace hyper-parameters
        self.enable_trace = config.enable_trace
        self.trace_coef = getattr(config, 'trace_coef', 0.05)
        self.unified_loss = getattr(config, 'loss_unified', False)
        self.future_weight = getattr(config, 'future_weight', 1.0)
        self.motion_weight = getattr(config, 'motion_weight', 1.0)
        self.motion_gating_enabled = getattr(
            config,
            'motion_gating_enabled',
            getattr(config, 'motion_grad_gate_enabled', True),
        )
        self.motion_gating_beta = getattr(
            config,
            'motion_gating_beta',
            getattr(config, 'motion_spike_beta', 1.0),
        )
        self.motion_gating_tau = getattr(
            config,
            'motion_gating_tau',
            getattr(config, 'motion_spike_tau', 1.0),
        )
        self.motion_gating_warmup_steps = getattr(
            config,
            'motion_gating_warmup_steps',
            getattr(config, 'motion_warmup_steps', getattr(config, 'warmup_steps', 1)),
        )
        self.latent_ema_decay = getattr(config, 'latent_ema_decay', 0.99)
        self.latent_ema = None
        self._pending_motion_gate_stats = None
        self.K_frames = getattr(config, 'K_frames', 3)
        self.align_layer = getattr(config, 'align_layer', 20)
        self.future_align_layer = getattr(config, 'future_align_layer', 14)
        self.motion_align_layer = getattr(config, 'motion_align_layer', 21)
        
        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")
        
        
        is_resume = hasattr(config, 'resume_from') and config.resume_from
        if is_resume:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')
        
        print('*'*20,transformer_path)
        modelswitch(transformer_path, is_train = True)
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            enable_trace=self.enable_trace,
            is_resume=is_resume,
            trace_dimension=getattr(config, 'trace_dimension', None),
            target_dim=getattr(config, 'target_dim', None),
        )
        # if not is_resumed:
        self.transformer._init_trace_parameters(
            # data_type = torch.float32,
            K_frames = self.K_frames,
            align_layer = self.align_layer,
            future_align_layer=self.future_align_layer,
            motion_align_layer=self.motion_align_layer,
        )
        
        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )

        self.transformer.train()
        self.transformer.requires_grad_(True)
        self.trainable_params = tuple(
            p for p in self.transformer.parameters() if p.requires_grad
        )
        is_aligning = False
        for name, _ in self.transformer.named_parameters():
            if 'motion' in name or 'trace' in name:
                is_aligning = True
        print('Is aligning......')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=1
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
            # collate_fn = collate_get_mask,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.layer_grad_norm_log_path = Path(config.save_root) / "dit_layer_grad_norms.jsonl"
        self.motion_gating_log_path = Path(config.save_root) / "motion_gating.jsonl"

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict, config):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5
        )
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0
        )

        # batch_dict['text_embed_real'] = batch_dict['text_emb']
        B, T, D = batch_dict['text_emb'].shape
        if T < config.max_tokens:
            batch_dict['text_emb'] = F.pad(
                batch_dict['text_emb'],
                (0, 0, 0, config.max_tokens - T),  # (D_left, D_right, T_left, T_right)
            )
        if B == 1:
            batch_dict['text_active_length'] = T
        if batch_dict['text_emb'].dtype != torch.bfloat16:
            batch_dict['text_emb'] = batch_dict['text_emb'].to(torch.bfloat16)
        if D != 4096:
            return False
        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        global FIRST
        if FIRST:
            for key in latent_dict:
                if isinstance(latent_dict[key],torch.Tensor) or isinstance(latent_dict[key],np.ndarray):
                    print(key, latent_dict[key].shape)
                else:
                    print(key, latent_dict[key])
            FIRST = False

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
            'text_active_length': batch_dict['text_active_length'],
        }

        if 'trace' in batch_dict:
            input_dict['trace'] = batch_dict['trace']
        
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def _scale_alignment_losses(self, alignment_loss, reference_loss, motion_scale=1.0):
        zero = reference_loss.new_zeros(())
        if not self.enable_trace or alignment_loss is None:
            return {'total': zero}

        # trace_coef, future_weight and motion_weight are the same meaning, don't repeat to mutiply .
        scale = 1 / self.gradient_accumulation_steps
        if isinstance(alignment_loss, (tuple, list)):
            if len(alignment_loss) != 2:
                raise ValueError(f"Expected alignment tuple as (future_loss, motion_loss), got {len(alignment_loss)} values")

            future_loss, motion_loss = alignment_loss
            future_loss = future_loss * self.future_weight * scale
            motion_loss = motion_loss * motion_scale * scale
            return {
                'total': future_loss + motion_loss,
                'future': future_loss,
                'motion': motion_loss,
            }
        else:
            alignment_loss = alignment_loss * self.trace_coef
            return {'total': alignment_loss * scale}

    def _append_alignment_losses(self, accumulated_align_losses, alignment_losses):
        for name, loss in alignment_losses.items():
            accumulated_align_losses.setdefault(name, []).append(loss.detach())

    def _compute_motion_gating_scale(self, latent_loss):
        if not self.motion_gating_enabled or not self.enable_trace:
            return self.motion_weight, None

        latent_loss_for_gate = latent_loss.detach().float()
        if dist.is_initialized():
            latent_loss_for_gate = dist_mean(latent_loss_for_gate)

        if self.latent_ema is None:
            self.latent_ema = latent_loss_for_gate.clone()

        warmup_steps = max(1, int(self.motion_gating_warmup_steps))
        motion_warmup = min(float(self.step) / float(warmup_steps), 1.0)
        spike_ratio = latent_loss_for_gate / (self.latent_ema + 1e-8)
        spike_scale = math.exp(
            -float(self.motion_gating_beta)
            * max(0.0, spike_ratio.detach().cpu().item() - float(self.motion_gating_tau))
        )
        motion_scale = self.motion_weight * motion_warmup * spike_scale

        self.latent_ema = (
            self.latent_ema * float(self.latent_ema_decay)
            + latent_loss_for_gate * (1.0 - float(self.latent_ema_decay))
        )
        stats = {
            'step': self.step,
            'latent_loss': latent_loss_for_gate.detach().cpu().item(),
            'latent_ema': self.latent_ema.detach().cpu().item(),
            'spike_ratio': spike_ratio.detach().cpu().item(),
            'spike_scale': spike_scale,
            'motion_warmup': motion_warmup,
            'base_motion_weight': self.motion_weight,
            'motion_scale': motion_scale,
        }
        return motion_scale, stats

    def _log_motion_gating(self, stats):
        if stats is None or self.config.rank != 0:
            return
        with self.motion_gating_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(stats) + '\n')

    def _collect_dit_layer_grad_norms(self):
        grad_norm_sqs = []
        grad_param_counts = []
        for block in self.transformer.blocks:
            block_norm_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            block_grad_count = torch.zeros((), device=self.device, dtype=torch.float32)
            for param in block.parameters():
                if param.grad is None:
                    continue
                grad = param.grad
                if hasattr(grad, "to_local"):
                    grad = grad.to_local()
                block_grad_count += 1
                block_norm_sq = block_norm_sq + grad.detach().float().pow(2).sum()
            grad_norm_sqs.append(block_norm_sq)
            grad_param_counts.append(block_grad_count)

        if not grad_norm_sqs:
            return []

        grad_norm_sqs = torch.stack(grad_norm_sqs)
        grad_param_counts = torch.stack(grad_param_counts)
        if dist.is_initialized():
            dist.all_reduce(grad_norm_sqs, op=dist.ReduceOp.SUM)
            dist.all_reduce(grad_param_counts, op=dist.ReduceOp.MAX)

        grad_norms = grad_norm_sqs.sqrt().detach().cpu().tolist()
        grad_counts = grad_param_counts.detach().cpu().tolist()
        return [
            {
                'layer': layer,
                'grad_norm': grad_norm,
                'grad_param_count': int(grad_count),
            }
            for layer, (grad_norm, grad_count) in enumerate(zip(grad_norms, grad_counts))
        ]

    def _should_log_layer_grad_norms(self, latent_loss, allow_periodic_log):
        unscaled_latent_loss = (latent_loss.detach() * self.gradient_accumulation_steps).float()
        if dist.is_initialized():
            log_latent_loss = dist_max(unscaled_latent_loss).detach().cpu().item()
        else:
            log_latent_loss = unscaled_latent_loss.detach().cpu().item()

        should_log_for_loss = log_latent_loss > 0.5
        if should_log_for_loss:
            return True, 'latent_loss_ge_0.5', log_latent_loss
        if allow_periodic_log and self.step % 20 == 0:
            return True, 'every_20_steps', log_latent_loss
        return False, None, log_latent_loss

    def _log_dit_layer_grad_norms(
        self,
        layer_grad_norms,
        reason,
        latent_loss,
        valid_batch_count,
    ):
        if self.config.rank != 0:
            return

        record = {
            'step': self.step,
            'reason': reason,
            'latent_loss': latent_loss,
            'valid_batch_count': valid_batch_count,
            'layers': layer_grad_norms,
        }
        with self.layer_grad_norm_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def _summarize_alignment_losses(self, accumulated_align_losses):
        summaries = {}
        for name, losses in accumulated_align_losses.items():
            if len(losses) == 0:
                continue

            stacked_sum = torch.stack(losses).sum()
            summaries[name] = {
                'avg': dist_mean(stacked_sum).detach().cpu().item(),
                'max': dist_max(stacked_sum).detach().cpu().item(),
            }

        if 'total' not in summaries:
            summaries['total'] = {'avg': 0, 'max': 0}

        return summaries
    
    def compute_loss(self,
        input_dict,
        pred,
    ):  
        alignment_loss = None
        if len(pred) == 3:
            latent_pred, action_pred, alignment_loss = pred
        else:
            latent_pred, action_pred = pred
        # print(alignment_loss)
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        motion_scale, motion_gate_stats = self._compute_motion_gating_scale(latent_loss)
        self._pending_motion_gate_stats = motion_gate_stats

        alignment_losses = self._scale_alignment_losses(
            alignment_loss,
            latent_loss,
            motion_scale=motion_scale,
        )
        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps, alignment_losses

    def _finalize_optimizer_step(
        self,
        accumulated_latent_losses,
        accumulated_action_losses,
        accumulated_align_losses,
        progress_bar,
        layer_grad_norms,
        layer_grad_norm_log_reason,
        layer_grad_norm_log_latent_loss,
        layer_grad_norm_log_valid_batch_count,
    ):
        num_accumulated_batches = len(accumulated_latent_losses)
        if layer_grad_norms:
            self._log_dit_layer_grad_norms(
                layer_grad_norms,
                layer_grad_norm_log_reason,
                layer_grad_norm_log_latent_loss,
                layer_grad_norm_log_valid_batch_count,
            )
        total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        lr = self.lr_scheduler.get_last_lr()[0]

        latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
        action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

        max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
        max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

        alignment_summaries = self._summarize_alignment_losses(accumulated_align_losses)
        alignment_loss_show = alignment_summaries['total']['avg']
        max_alignment_loss_show = alignment_summaries['total']['max']

        torch.cuda.synchronize()
        if self.step % self.config.gc_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if self.config.rank == 0:
            progress_bar.n += num_accumulated_batches
            postfix = {
                'latent_loss': f'{latent_loss_show:.5f}',
                'action_loss': f'{action_loss_show:.5f}',
                'alignment_loss': f'{alignment_loss_show:.5f}',
                'step': self.step,
                'grad_norm': f'{total_norm.item():.3f}',
                'lr': f'{lr:.2e}'
            }
            if 'future' in alignment_summaries:
                postfix['future_align'] = f"{alignment_summaries['future']['avg']:.5f}"
            if 'motion' in alignment_summaries:
                postfix['motion_align'] = f"{alignment_summaries['motion']['avg']:.5f}"
            progress_bar.set_postfix(postfix)

            if self.config.enable_wandb:
                wandb_metrics = {
                    'loss_metrics/global_avg_video_loss': latent_loss_show,
                    'loss_metrics/global_avg_action_loss': action_loss_show,
                    'loss_metrics/global_avg_alignment_loss': alignment_loss_show,
                    'loss_metrics/global_max_video_loss': max_latent_loss_show,
                    'loss_metrics/global_max_action_loss': max_action_loss_show,
                    'loss_metrics/global_max_alignment_loss': max_alignment_loss_show,
                    'grad_norm': total_norm.item(),
                    'lr': lr,
                }
                for name, summary in alignment_summaries.items():
                    if name == 'total':
                        continue
                    wandb_metrics[f'loss_metrics/global_avg_alignment_{name}_loss'] = summary['avg']
                    wandb_metrics[f'loss_metrics/global_max_alignment_{name}_loss'] = summary['max']
                self.wandb.log(wandb_metrics, step=self.step)

        self.step += 1
        if self.step % self.config.save_interval == 0:
            if self.config.rank == 0:
                logger.info(f"Starting save model at step {self.step}")
            self.save_checkpoint()

    def _run_train_micro_step(
        self,
        input_dict,
        valid_batch_count,
        accumulated_latent_losses,
        accumulated_action_losses,
        accumulated_align_losses,
        progress_bar,
        is_last_valid_batch=False,
    ):
        should_sync = (
            (valid_batch_count + 1) % self.gradient_accumulation_steps == 0
            or is_last_valid_batch
        )
        self.transformer.set_requires_gradient_sync(should_sync)

        align_modules = {
            'dynamic': motion_incremental_alignment_tokenwise,
            'future': future_alignment_loss,
        }
        output = self.transformer(input_dict, alignment_modules = align_modules, train_mode = True)
        latent_loss, action_loss, alignment_losses = self.compute_loss(input_dict, output)
        self._log_motion_gating(self._pending_motion_gate_stats)
        
        loss = latent_loss + action_loss + alignment_losses['total']

        layer_grad_norms = []
        layer_grad_norm_log_reason = None
        layer_grad_norm_log_latent_loss = None
        layer_grad_norm_log_valid_batch_count = valid_batch_count
        should_log_grad_norms, layer_grad_norm_log_reason, layer_grad_norm_log_latent_loss = (
            self._should_log_layer_grad_norms(latent_loss, allow_periodic_log=should_sync)
        )

        self.transformer.set_requires_gradient_sync(should_sync)
        loss.backward()

        if should_log_grad_norms:
            layer_grad_norms = self._collect_dit_layer_grad_norms()
            if not should_sync:
                self._log_dit_layer_grad_norms(
                    layer_grad_norms,
                    layer_grad_norm_log_reason,
                    layer_grad_norm_log_latent_loss,
                    layer_grad_norm_log_valid_batch_count,
                )
                layer_grad_norms = []

        accumulated_latent_losses.append(latent_loss.detach())
        accumulated_action_losses.append(action_loss.detach())
        self._append_alignment_losses(accumulated_align_losses, alignment_losses)

        if should_sync:
            self._finalize_optimizer_step(
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
                progress_bar,
                layer_grad_norms,
                layer_grad_norm_log_reason,
                layer_grad_norm_log_latent_loss,
                layer_grad_norm_log_valid_batch_count,
            )
            accumulated_latent_losses = []
            accumulated_action_losses = []
            accumulated_align_losses = {}

        return (
            valid_batch_count + 1,
            accumulated_latent_losses,
            accumulated_action_losses,
            accumulated_align_losses,
        )

    def train_epoch(self):
        self.transformer.train()

        # Use manual progress bar control to only update on optimizer steps
        progress_bar = tqdm(
            total=len(self.train_loader),
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True, 
            dynamic_ncols=True
        )

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_align_losses = {}
        valid_batch_count = 0
        pending_input_dict = None
        for batch in self.train_loader:
            batch = self.convert_input_format(batch)

            input_dict = self._prepare_input_dict(batch, self.config)
            if isinstance(input_dict,bool) and not input_dict:
                continue

            if pending_input_dict is not None:
                (
                    valid_batch_count,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_align_losses,
                ) = self._run_train_micro_step(
                    pending_input_dict,
                    valid_batch_count,
                    accumulated_latent_losses,
                    accumulated_action_losses,
                    accumulated_align_losses,
                    progress_bar,
                    is_last_valid_batch=False,
                )

            pending_input_dict = input_dict

        if pending_input_dict is not None:
            (
                valid_batch_count,
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
            ) = self._run_train_micro_step(
                pending_input_dict,
                valid_batch_count,
                accumulated_latent_losses,
                accumulated_action_losses,
                accumulated_align_losses,
                progress_bar,
                is_last_valid_batch=True,
            )

        progress_bar.close()

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")

        while self.step < self.config.num_steps:
            self.train_epoch()
            if dist.is_initialized():
                dist.barrier()

        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name] # datasets

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f'-------world_size:{world_size}---------')
    if world_size > 1:
        print('world_size, local_rank, rank',world_size, local_rank, rank)
        init_distributed(world_size, local_rank, rank)
    else:
        # 单进程：确保当前卡设置好
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    # init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.disable_motion_grad_gate:
        config.motion_grad_gate_enabled = False
    if args.motion_gating_beta is not None:
        config.motion_gating_beta = args.motion_gating_beta
    if args.motion_gating_tau is not None:
        config.motion_gating_tau = args.motion_gating_tau
    if args.motion_gating_warmup_steps is not None:
        config.motion_gating_warmup_steps = args.motion_gating_warmup_steps

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
    # pdb.set_trace()
    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--disable-motion-grad-gate",
        action="store_true",
        help="Disable motion gating.",
    )
    parser.add_argument(
        "--motion-gating-beta",
        type=float,
        default=None,
        help="Beta for exp(-beta * max(0, spike_ratio - tau)).",
    )
    parser.add_argument(
        "--motion-gating-tau",
        type=float,
        default=None,
        help="Tau threshold for latent-loss spike gating.",
    )
    parser.add_argument(
        "--motion-gating-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for motion gating scale.",
    )
    parser.add_argument(
        "--motion-grad-gate-threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--motion-grad-gate-conflict-scale",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":

    init_logger()
    # mp.set_start_method("spawn", force=True)
    main()
