import torch
import os
import numpy as np

file_path = '/data/.cache/datasets/lerobot/robotwin/tiny/blocks_ranking_size/'
keys = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
latent_path = os.path.join(file_path,'latents')
trace_path = os.path.join(file_path,'trace')
parent_path_for_latent = os.path.join(latent_path, 'chunk-000')
# sample = os.path.join(latent_path,'chunk-000/observation.images.cam_high/episode_000000_0_276.pth')
# for s in os.listdir(parent_path_for_latent):
#     sample = os.path.join(parent_path_for_latent,s)

#     print(sample)
#     data = torch.load(sample)
#     for key in data:
#         if isinstance(data[key],torch.Tensor) or isinstance(data[key],np.ndarray):
#             print(key, data[key].shape)
#         else:
#             print(key, data[key])

for key in keys:
    cam_path = os.path.join(parent_path_for_latent,key)
    sample = os.path.join(cam_path, os.listdir(cam_path)[0])
    data = torch.load(sample)
    for key in data:
        if isinstance(data[key],torch.Tensor) or isinstance(data[key],np.ndarray):
            print(key, data[key].shape)
        else:
            print(key, data[key])

for key in keys:
    cam_path = os.path.join(trace_path, 'chunk-000',key)
    sample = os.path.join(cam_path, os.listdir(cam_path)[0])
    data = torch.load(sample)
    print(f'{sample} shape: {data.shape}')

# print(data['video_num_frames'])

#dict_keys(['latent', 'latent_num_frames', 'latent_height', 'latent_width', 'video_num_frames', 
# 'video_height', 
# 'video_width', 'text_emb', 'text', 'frame_ids', 'start_frame', 'end_frame', 'fps', 'ori_fps'])

'''
latent torch.Size([5760, 48])
latent_num_frames 18
latent_height 16
latent_width 20
video_num_frames 69
video_height 256
video_width 320
text_emb torch.Size([512, 4096])
text Open the drawer of the smooth-surface cabinet and move the rectangular soap with slight groove pattern into it.
frame_ids [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272]
start_frame 0
end_frame 276
fps 12
ori_fps 50
'''

'''
timesteps torch.Size([1, 8])
noisy_latents torch.Size([1, 48, 8, 24, 20])
targets torch.Size([1, 48, 8, 24, 20])
latent torch.Size([1, 48, 8, 24, 20])
cond_timesteps torch.Size([1, 8])
grid_id torch.Size([1, 4, 960])
text_emb torch.Size([1, 512, 4096])
'''


'''
(8,10)
(8,10)
(16,20)
'''