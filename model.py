import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    '''Base network for student and teacher

    Args:
        pretrained (bool): whether to get the pretrained version of resnet18
            Defaults to `False`
    '''

    def __init__(self, pretrained=False) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.res = resnet18(weights=weights)

        # register hooks to get intermediate feature maps
        self.maps = []
        self._register_hooks()
        
    def _register_hooks(self):
        '''Register forward hooks on conv2_x, conv3_x and conv4_x layers'''
        def hook(module, input, output):
            self.maps.append(output)
        layers = [child for name, child in self.res.named_children() if name.startswith('layer')]

        # register hooks on layers conv2_x, conv3_x and conv4_x
        for layer in layers[:-1]:
            layer[-1].register_forward_hook(hook)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        '''Performs a forward pass through the network

        Args:
            x (torch.Tensor): input batch

        Returns:
            self.maps (list[torch.Tensor]): intermediate feature maps
        '''
        self.maps = []
        _ = self.res(x)
        return self.maps

    
class STFPM(nn.Module):
    '''Student-teacher feature pyramid matching
    
    Args:
        size (int): input size
    '''
    
    def __init__(self, size: int) -> None:
        super().__init__()

        # initialize teacher and student
        self.teacher = ResNet(pretrained=True)
        self.student = ResNet(pretrained=False)

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
        anomaly_map = torch.ones((b, 256, 256), device=x.device)

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