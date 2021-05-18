import numpy as np
import torch

def build_mask_np(observation, pen=1):
    green_mask = np.zeros_like(observation[:, :, 0])
    green_mask[observation[:,:, 1] > 200] = 1
    red_mask = np.zeros_like(observation[:, :, 0])
    red_mask[observation[:, :, 0] > 220] = 1
    
    if pen > 1:
      for mask in [green_mask, red_mask]:
          mask *= 1000
          mask += 1
    return green_mask, red_mask
def build_mask(observation):
    green_mask = torch.zeros_like(observation[0])
    green_mask[observation[1,:, :] > -0.3] = 1
    
    red_green_mask = torch.zeros_like(observation[0])
    red_green_mask[observation[0,:, :] < 0.2] = 1
    
    ghost_mask = green_mask * red_green_mask

    red_mask = torch.zeros_like(observation[0])
    red_mask[observation[0,:, :] > 0.55] = 1
    
    boxes_mask = green_mask * red_mask

    for mask in [ghost_mask, boxes_mask]:
        mask *= 10
        mask += 1
    return ghost_mask, boxes_mask

def make_small_objects_important(reconstructed, image, only_ghost=True):
    diff = (reconstructed - image)

    for idx, observation in enumerate(image):
        ghost_mask, boxes_mask = build_mask(observation)
        
        masks_to_apply = ghost_mask.clone()
        if not only_ghost:
          print("should not print this")
          masks_to_apply += boxes_mask

        for ch_id in range(3):
          diff[idx, ch_id, :, :] = diff[idx, ch_id, :, :] * masks_to_apply
        
    diff = diff **2
    return torch.mean(diff)

