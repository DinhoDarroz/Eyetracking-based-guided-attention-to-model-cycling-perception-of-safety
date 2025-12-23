from torch.nn.init import trunc_normal_ as _torch_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Thin wrapper so DINO's 'from utils import trunc_normal_' works.

    Matches the signature of torch.nn.init.trunc_normal_ and is close enough
    to DINO's own implementation for practical purposes.
    """
    return _torch_trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
