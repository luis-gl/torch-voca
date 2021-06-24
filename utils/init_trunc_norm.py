'''
Código extraído de https://gist.github.com/Enealor/c2308bc67fc7646cba086cae267b6d69
que implementa el muestreo truncado en la distribución normal
'''

import math
import torch

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    #Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        #Computes standard normal cumulative distribution function
        return (1+math.erf(x/math.sqrt(2.)))/2.
    
    with torch.no_grad():
        #Compute upper and lower cdf values
        l = norm_cdf((a-mean)/std)
        u = norm_cdf((b-mean)/std)
        sqrt_two = math.sqrt(2.)
        
        #First, fill with uniform values
        tensor.uniform_(0., 1.)

        #Scale by 2(u-l), shift by 2l-1
        tensor.mul_(2*(u-l))
        tensor.add_(2*l-1)
        
        #Ensure that the values are strictly between -1 and 1
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.-eps), max=(1.-eps))

        #Now use the inverse erf to get distributed values
        tensor.erfinv_()

        #Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input `Tensor` with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)