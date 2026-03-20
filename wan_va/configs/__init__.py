# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_infer_cfg import va_robotwin_infer_cfg
from .va_robotwin_train_task10_cfg import va_robotwin_train_cfg as va_robotwin_train_task10_cfg
from .va_robotwin_infer_task10_cfg import va_robotwin_train_cfg as va_robotwin_infer_task10_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'robotwin_infer': va_robotwin_infer_cfg,
    'robotwin_train_task10': va_robotwin_train_task10_cfg,
    'robotwin_infer_task10': va_robotwin_infer_task10_cfg,
}