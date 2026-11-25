"""Convolutional autoencoder and its original adaptation variants."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, mlp=False):
        super().__init__(); self.mlp = mlp
        if mlp:
            self.encoder = nn.Sequential(nn.Linear(386,128),nn.GELU(),nn.Linear(128,32),nn.GELU())
            self.decoder = nn.Sequential(nn.Linear(32,128),nn.GELU(),nn.Linear(128,384))
        else:
            self.encoder = nn.Sequential(nn.Conv2d(386,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU(),
                                         nn.Conv2d(128,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU())
            self.decoder = nn.Sequential(nn.Conv2d(32,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU(),
                                         nn.Conv2d(128,384,3,padding=1))

    def forward(self, x, corrupt=False):
        b, _, h, w = x.shape
        if corrupt: x = x * (torch.rand_like(x) >= .1)
        yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=x.device),
                                torch.linspace(-1,1,w,device=x.device), indexing='ij')
        x = torch.cat([x,torch.stack([xx,yy])[None].expand(b,-1,-1,-1)],1)
        if self.mlp: x = x.permute(0,2,3,1)
        x = self.decoder(self.encoder(x))
        return x.permute(0,3,1,2) if self.mlp else x

    def adapt_parameters(self, mode):
        self.requires_grad_(False)
        if mode == 'norm':
            for layer in self.modules():
                if isinstance(layer, nn.GroupNorm): layer.requires_grad_(True)
        else: self.decoder.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def score(self, x):
        self.eval(); anomaly = (self(x)-x).square().mean(1)
        score = anomaly.flatten(1).topk(max(1,math.ceil(.01*anomaly[0].numel())),1).values.mean(1)
        return score, anomaly


def optimize(model, source, validation, target, steps, lr, seed, mode='source', callback=None):
    """Microbatch one; effective source batch two or exactly 1:1 target/replay."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    params = list(model.parameters()) if mode == 'source' else model.adapt_parameters(mode)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    device = next(model.parameters()).device
    best, best_state, curve = float('inf'), None, []
    counts = {'source':0, 'target':0}; model.train()
    for step in range(1, steps+1):
        if mode == 'source': samples = [('source',source)]*2
        elif mode == 'no_replay': samples = [('target',target)]
        else: samples = [('target',target),('source',source)]
        optimizer.zero_grad(set_to_none=True); loss_sum = 0.
        for name, pool in samples:
            x = pool[int(rng.integers(len(pool)))].to(device)
            loss = F.mse_loss(model(x,corrupt=True),x)/len(samples)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); loss_sum += loss.item(); counts[name] += 1
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        if mode == 'source' and (step % 100 == 0 or step == steps):
            model.eval()
            with torch.no_grad():
                val = np.mean([F.mse_loss(model(x.to(device)),x.to(device)).item() for x in validation])
            curve.append({'step':step,'train_loss':loss_sum,'val_loss':float(val)})
            if val < best:
                best = val; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.train()
        if callback: callback(step)
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return {'curve':curve,'exposures':counts,'steps':steps,'trainable_parameters':sum(p.numel() for p in params)}
