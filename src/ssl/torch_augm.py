import torch

class FlipAug:

    def __init__(self, dims):
        self.dims = dims
        
    def direct(self, x):
        return torch.flip(x, self.dims)
    
    def reverse(self, x):
        return torch.flip(x, self.dims)

class NoiseAug:
    def __init__(self, scale):
        self.scale = scale

    def direct(self, x):
        noise = torch.randn_like(x, device=x.device)*self.scale
        return x + noise

    def reverse(self, x):
        return x

class BrightnessAug:
    def __init__(self, scale):
        self.scale = scale

    def direct(self, x):
        scale = (torch.rand((1), device=x.device) - 0.5)*self.scale
        additive = torch.ones_like(x, device=x.device)*scale
        return x + additive

    def reverse(self,x):
        return x