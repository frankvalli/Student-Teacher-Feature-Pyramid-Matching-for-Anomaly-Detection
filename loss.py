import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    '''Distillation loss
    
    Args:
        alphas (list[float]): weights for feature maps
    '''
    def __init__(self, alphas: list[float]) -> None:
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.scale = alphas
 
    def forward(self, t_maps, s_maps) -> torch.Tensor:
        '''Calculate loss

        Args:
            t_maps (list[torch.Tensor]): teacher feature maps
            s_maps (list[torch.Tensor]): student feature maps

        Returns:
            total_loss (torch.Tensor): loss
        '''
        total_loss = 0.

        for idx, (t_map, s_map) in enumerate(zip(t_maps, s_maps)):
            # normalize feature maps
            t_map_norm = F.normalize(t_map, dim=-3)
            s_map_norm = F.normalize(s_map, dim=-3)
            
            # compute mse loss
            loss = self.mse(t_map_norm, s_map_norm)
            
            # scale loss and add to the total loss
            _, _, h, w = t_map.shape
            total_loss += self.scale[idx] * loss / (2 * h * w)

        return total_loss