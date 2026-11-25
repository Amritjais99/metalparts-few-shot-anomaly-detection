"""PatchCore memory variant and its pinned CNN assets."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ASSETS = Path(__file__).resolve().parent / "assets"
WEIGHTS = ASSETS / "wide_resnet50_2-95faca4d.pth"


class Memory:
    def __init__(self, cosine, cfg):
        self.cosine, self.cfg, self.bank = cosine, cfg, None

    def fit(self, source, target, device, alignment=False):
        """Pools are (ID, CPU feature tensor). Stable per-image sampling and a fixed total bank."""
        delta = None
        if alignment:
            delta = torch.stack([x.mean((0,2,3)) for _,x in target]).mean(0) - \
                    torch.stack([x.mean((0,2,3)) for _,x in source]).mean(0)
        patches, owners, identities = [], [], []
        for is_target, pool in ((False,source),(True,target)):
            for identity, x in pool:
                vectors = x[0].flatten(1).T.clone()
                if delta is not None and not is_target: vectors += delta
                if self.cosine: vectors = F.normalize(vectors,dim=1)
                seed = int(hashlib.sha256(('12345:'+identity).encode()).hexdigest()[:16],16)
                rng = np.random.default_rng(seed)
                index = rng.choice(len(vectors), min(128,len(vectors)), replace=False)
                patches.append(vectors[index]); owners.extend([int(is_target)]*len(index)); identities.extend([identity]*len(index))
        candidates = torch.cat(patches); owners = np.asarray(owners); identities = np.asarray(identities)
        rng = np.random.default_rng(12345)
        if len(candidates) > 8192:
            index = rng.choice(len(candidates),8192,replace=False)
            candidates, owners, identities = candidates[index], owners[index], identities[index]
        candidates = candidates.to(device)
        generator = torch.Generator(device=device).manual_seed(12345)
        projection = torch.randn(candidates.shape[1],64,device=device,generator=generator)/8
        reduced = candidates @ projection
        first = (reduced-reduced.mean(0)).square().sum(1).argmax()
        nearest = torch.full((len(reduced),),float('inf'),device=device)
        selected = []; index = first
        for _ in range(min(2048,len(candidates))):
            selected.append(index)
            nearest = torch.minimum(nearest,(reduced-reduced[index]).square().sum(1))
            nearest[index] = -1
            index = nearest.argmax()
        selected = torch.stack(selected)
        self.bank = candidates[selected].contiguous()
        ids, counts = np.unique(identities[selected.cpu().numpy()],return_counts=True)
        self.info = {'bank_size':len(selected),'target_bank_entries':int(owners[selected.cpu().numpy()].sum()),
                     'bank_owners':dict(zip(ids.tolist(),counts.tolist())),
                     'trainable_parameters':0,'alignment':alignment}
        return self.info

    @torch.no_grad()
    def score(self,x):
        _, _, h, w = x.shape; vectors = x[0].flatten(1).T
        if self.cosine: vectors = F.normalize(vectors,dim=1)
        scores = []
        for chunk in vectors.split(256):
            if self.cosine: distance = (1-chunk @ self.bank.T).clamp_min(0)
            else: distance = (chunk.square().sum(1)[:,None]+self.bank.square().sum(1)[None]-2*chunk@self.bank.T).clamp_min(0)
            scores.append(distance.min(1).values)
        anomaly = torch.cat(scores).reshape(1,h,w)
        if self.cosine: score = anomaly.flatten().topk(math.ceil(.01*h*w)).values.mean().reshape(1)
        else: score = anomaly.max().reshape(1)
        return score, anomaly
