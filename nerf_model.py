import torch
from torch import nn

class NerfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(60, 256), nn.ReLU(),
            *[*[nn.Linear(256, 256), nn.ReLU()] * 4],
        )
        self.block2 = nn.Sequential(
            nn.Linear(60 + 256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 + 1)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256 + 24, 128), 
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )


    def _positional_encoding(self, position, direction):
        L_1 = 10
        L_2 = 4

        position = [func((2 ** i) * torch.pi * position) for i in range(L_1) for func in [torch.sin, torch.cos]]
        direction = [func((2 ** i) * torch.pi * direction) for i in range(L_2) for func in [torch.sin, torch.cos]]

        position = torch.concat(position, dim=0)
        direction = torch.concat(direction, dim=0)
        # return position.to("cuda"), direction.to("cuda")
        return position, direction


    def forward(self, position, direction):
        position, direction = self._positional_encoding(position, direction)
        position, direction = position.permute((1, 2, 3, 0)), direction.permute((1, 2, 3, 0))
        x = self.block1(position)
        x = torch.cat([position, x], dim=-1)

        x = self.block2(x)
        x, sigma = x[..., :-1], x[..., -1]
        sigma = torch.max(torch.tensor(1e-6), sigma)

        x = torch.cat([x, direction], dim=-1)

        x = self.block3(x)
        return x, sigma