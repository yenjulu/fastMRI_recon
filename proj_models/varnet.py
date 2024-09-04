"""
Variational Network

Reference:
* Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. Learning a variational network for reconstruction of accelerated MRI data. Magn Reson Med 2018;79:3055-3071.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from utils import r2c, c2r
from proj_models import mri, unet

# %%
class data_consistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self,
                curr_x: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        A = mri.SenseOp(coil, mask)
        grad = A.adj(A.fwd(curr_x)) - x0

        next_x = curr_x - self.lam * grad   # equation[3] in paper

        return next_x

# %%
class VarNet(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        self.dc = data_consistency()
        self.dws = nn.ModuleList([unet.Unet(2, 2, num_pool_layers=n_layers) for _ in range(self.n_cascades)])
        
    def forward(self, x0, coil, mask):
        x0 = r2c(x0, axis=1)

        for c in range(self.n_cascades):
            if c == 0: 
                x_prev = x0.clone()
                
            z = self.dc(x_prev, x0, coil, mask)
            u = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            x = z - u 
            x_prev = x

        return x, u # c2r(x, axis=1), c2r(u, axis=1)