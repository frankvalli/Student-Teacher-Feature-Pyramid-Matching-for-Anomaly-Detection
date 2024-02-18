import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
    

class ResNet(nn.Module):
    '''Base network for student and teacher

    Args:
        layers (list[str]): layers to get feature maps from
        pretrained (bool): whether to get the pretrained version of resnet18
    '''

    def __init__(self, layers: list[str], pretrained: bool) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.res = resnet18(weights=weights)
        self.layers = layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        '''Performs a forward pass through the network

        Args:
            x (torch.Tensor): input batch

        Returns:
            self.maps (list[torch.Tensor]): intermediate feature maps
        '''
        maps = []
        for name, module in list(self.res.named_children())[:-2]:
            x = module(x)         
            if name in self.layers:
                maps.append(x)
        return maps


class STFPM(nn.Module):
    '''Student-teacher feature pyramid matching
    
    Args:
        layers (list[str]): layers to get feature maps from
        size (int): input size
    '''
    
    def __init__(self, layers: list[str], size: int) -> None:
        super().__init__()

        # initialize teacher and student
        self.teacher = ResNet(layers=layers, pretrained=True)
        self.student = ResNet(layers=layers, pretrained=False)

        # instantiate upsample
        self.size = size
        self.upsample = nn.Upsample(size=(self.size, self.size), mode='bilinear')

        # freeze teacher parameters
        for parameters in self.teacher.parameters():
            parameters.requires_grad = False

    def get_anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        '''Get anomaly maps

        Args:
            x (torch.Tensor): input batch

        Returns:
            anomaly_map (torch.Tensor): final anomaly map
            anomaly_map_list (list[torch.Tensor]): intermediate anomaly maps
        '''
        b = x.shape[0]
        t_maps, s_maps = self.forward(x)

        anomaly_map_list = []
        anomaly_map = torch.ones((b, self.size, self.size), device=x.device)

        for t_map, s_map in zip(t_maps, s_maps):
            # normalize feature maps
            t_map_norm = F.normalize(t_map, dim=-3)
            s_map_norm = F.normalize(s_map, dim=-3)

            # calculate anomaly map
            am = 0.5 * torch.sum((t_map_norm - s_map_norm) ** 2, dim=-3)
            am = self.upsample(am.unsqueeze(1))
            am = am.squeeze(1)
            anomaly_map_list.append(am)
            anomaly_map *= am
        
        return anomaly_map, anomaly_map_list

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        '''Perform a forward pass through the network

        Args:
            x (torch.Tensor): input batch

        Returns:
            self.teacher (list): intermediate feature maps of teacher
            self.student (List[torch.Tensor]): intermediate feature maps of student
        '''
        return self.teacher(x), self.student(x)