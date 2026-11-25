"""Label-free manifests, separation, pinned frozen features, atomic I/O."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from models.dinov2.model import REPOSITORY as DINO_REPOSITORY, WEIGHTS as DINO_WEIGHTS
from models.patchcore.model import WEIGHTS as CNN_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    cfg['config_hash'] = digest(path)
    cfg['protocol_hash'] = digest(ROOT / 'docs/protocol.md')
    cfg['code_hash'] = hashlib.sha256(''.join(digest(ROOT/n) for n in
        ('common/data.py', 'common/model.py', 'common/run.py', 'common/analyze.py', 'models/conv_ae/model.py', 'models/patchcore/model.py', 'models/dinov2/model.py')).encode()).hexdigest()
    base = ROOT / cfg['manifests']
    paths = [base/'shared/identity.csv'] + list(base.glob('*/*_roles.csv')) + list(base.glob('*/*_support_calibration_draws.csv'))
    cfg['manifest_hash'] = hashlib.sha256(''.join(digest(p) for p in sorted(paths,key=lambda p:p.name)).encode()).hexdigest()
    cfg['fingerprint'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg['raw'] = os.environ.get('DATA_DIR',cfg['raw'])
    return cfg


def check_separation(roles):
    """Evaluation-only dependence is disclosed; every learning/calibration role is separate."""
    evaluation = {'source_test', 'target_test'}
    for key in ('id', 'image_path', 'sha256', 'pixel_sha256', 'component'):
        if key not in roles: raise ValueError(f'Missing separation field: {key}')
        if roles[key].isna().any() or (roles[key].astype(str) == '').any():
            raise ValueError(f'Unknown identity field: {key}')
        for value, rows in roles.groupby(key):
            names = set(rows.role)
            if len(names) > 1 and not names <= evaluation:
                raise ValueError(f'LEAKAGE {key}={value}: {sorted(names)}')
    if roles.duplicated(['id', 'role']).any(): raise ValueError('Repeated manifest record')
    return True


def load_roles(cfg, group, condition, k, seed, controlled=False):
    base = ROOT / cfg['manifests']
    roles = pd.read_csv(base / group / f'{group}_roles.csv', keep_default_na=False)
    roles = roles[roles.condition == condition].copy()
    if roles.empty: raise ValueError('Unknown condition')
    identities = pd.read_csv(base / 'shared/identity.csv', keep_default_na=False)
    roles = roles.merge(identities[['id', 'pixel_sha256']], on='id', validate='many_to_one')
    draws = pd.read_csv(base / group / f'{group}_support_calibration_draws.csv')
    arm = 'controlled_10_plus_10' if controlled else 'all_k_adaptation'
    draw = draws[(draws.condition == condition) & (draws.seed == seed) &
                 (draws.k == k) & (draws.arm == arm)].sort_values('rank')
    if len(draw) != k: raise ValueError(f'Expected {k} support draw rows, got {len(draw)}')
    pool = roles[roles.role == 'target_support_pool'].set_index('id', drop=False)
    if not set(draw.id) <= set(pool.index): raise ValueError('Support outside authorized normal pool')
    selected = pool.loc[draw.id].copy().reset_index(drop=True)
    selected['role'] = draw.role.to_numpy()
    roles = pd.concat([roles[roles.role != 'target_support_pool'], selected], ignore_index=True)
    check_separation(roles)
    if controlled and (sum(roles.role == 'adaptation') != 10 or sum(roles.role == 'calibration') != 10):
        raise ValueError('Controlled comparison requires disjoint 10+10')
    return roles


def subset(roles, name):
    return roles[roles.role == name].sort_values('id').to_dict('records')


def verify_assets(cfg):
    for name, expected in cfg['asset_hashes'].items():
        if digest({p.name:p for p in (DINO_WEIGHTS,CNN_WEIGHTS)}[name]) != expected: raise ValueError(f'Asset changed: {name}')
    repo = DINO_REPOSITORY
    tree = hashlib.sha256(''.join(p.relative_to(repo).as_posix()+digest(p)
        for p in sorted(repo.rglob('*.py'),key=lambda p:p.relative_to(repo).as_posix())).encode()).hexdigest()
    if tree != cfg['repo_tree_hash']: raise ValueError('DINOv2 source tree changed')


class Features:
    def __init__(self, cfg, backbone, device, cache=None):
        self.cfg, self.backbone, self.device = cfg, backbone, torch.device(device)
        signature = {'backbone': backbone, 'resolution': cfg['resolution'], 'policy': 'rgb-pil-bilinear-imagenet-v1',
                     'assets': cfg['asset_hashes'], 'repo': cfg['repo_tree_hash'], 'precision': 'fp32-to-fp16',
                     'extractor_code':digest(Path(__file__)), 'torch':torch.__version__}
        self.key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.folder = Path(cache or os.environ.get('CACHE_DIR',str(ROOT/'cache'))) / self.key
        self.folder.mkdir(parents=True, exist_ok=True)
        self.net = None

    def load(self):
        if self.net is not None: return
        verify_assets(self.cfg)
        if self.backbone == 'dino':
            os.environ['XFORMERS_DISABLED'] = '1'
            self.net = torch.hub.load(str(DINO_REPOSITORY), 'dinov2_vits14',
                                      source='local', pretrained=False)
            weights = DINO_WEIGHTS
        else:
            from torchvision.models import wide_resnet50_2
            self.net = wide_resnet50_2(weights=None)
            weights = CNN_WEIGHTS
        self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        if self.backbone == 'cnn':
            self.net = torch.nn.Sequential(*list(self.net.children())[:7])
        self.net.eval().requires_grad_(False).to(self.device)

    def get(self, row):
        path = self.folder / f"{row['id']}-{row['sha256'][:16]}.npy"
        if path.exists():
            array = np.load(path, allow_pickle=False)
        else:
            if os.environ.get('CACHE_READ_ONLY')=='1': raise ValueError('Required feature missing from completed extraction')
            raw = ROOT / self.cfg['raw'] / row['image_path']
            if digest(raw) != row['sha256']: raise ValueError('Raw image checksum changed')
            self.load()
            with Image.open(raw) as im:
                im = im.convert('RGB').resize((self.cfg['resolution'],)*2, Image.Resampling.BILINEAR)
                x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1)[None]/255
            x = (x - torch.tensor([.485,.456,.406])[None,:,None,None]) / torch.tensor([.229,.224,.225])[None,:,None,None]
            x = x.to(self.device)
            if self.backbone == 'dino':
                z = self.net.forward_features(x)['x_norm_patchtokens']
                side = self.cfg['resolution']//14
                z = F.normalize(z, dim=-1).transpose(1,2).reshape(1,384,side,side)
            else:
                maps = []
                for i, layer in enumerate(self.net):
                    x = layer(x)
                    if i in (5,6): maps.append(x)
                height, width = maps[0].shape[-2:]; patches = []
                for m in maps:
                    u = F.unfold(m, 3, padding=1).reshape(1,-1,*m.shape[-2:])
                    u = F.interpolate(u, (height,width), mode='bilinear', align_corners=False)
                    u = u.permute(0,2,3,1).reshape(-1,1,u.shape[1])
                    patches.append(F.adaptive_avg_pool1d(u,1024).squeeze(1))
                z = F.adaptive_avg_pool1d(torch.stack(patches,1).reshape(-1,1,2048),1024)
                z = z.reshape(1,height,width,1024).permute(0,3,1,2)
            array = z[0].float().cpu().numpy().astype(np.float16)
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as f: np.save(f, array, allow_pickle=False)
            temp.replace(path)
        expected = (384,32,32) if self.backbone == 'dino' else (1024,56,56)
        if array.shape != expected or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError(f'Invalid feature cache: {path.name}')
        return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)

    def close(self):
        self.net = None
        if self.device.type == 'cuda': torch.cuda.empty_cache()


def completed(path, fingerprint):
    path = Path(path)
    if not path.exists(): return False
    record = json.loads(path.read_text())
    if record['fingerprint'] != fingerprint: raise ValueError('Stale output fingerprint; do not overwrite')
    if record['status'] != 'complete': return False
    for item in record['files']:
        if digest(path.parent/item['path']) != item['sha256']: raise ValueError('Incomplete/corrupt output')
    return True


def selftest():
    import copy, tempfile, argparse
    from models.conv_ae.model import Autoencoder, optimize
    from models.dinov2.model import Memory
    from common.run import experiment
    from common.analyze import auc, aupro, integrate_pro, paired_inference
    from scipy import ndimage
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2); torch.manual_seed(7); checks={}
    x=F.normalize(torch.randn(1,384,32,32),dim=1); model=Autoencoder()
    assert model(x).shape==x.shape
    result=optimize(model,[x],[x],[],2,1e-3,7); checks['finite_loss_gradients_32_grid']=True
    for mode in ('decoder','norm','no_replay'):
        test=copy.deepcopy(model); before={k:v.clone() for k,v in test.named_parameters()}
        result=optimize(test,[x],[x],[x*.9],2,1e-4,8,mode)
        changed={k for k,v in test.named_parameters() if not torch.equal(before[k],v)}
        allowed={k for k,v in test.named_parameters() if v.requires_grad}
        assert changed and changed<=allowed
        if mode=='decoder': assert all(k.startswith('decoder.') for k in changed)
        if mode=='norm': assert changed <= {'encoder.1.weight','encoder.1.bias','encoder.4.weight','encoder.4.bias','decoder.1.weight','decoder.1.bias'}
        assert result['exposures']=={'source':0 if mode=='no_replay' else 2,'target':2}
        checks[mode+'_freeze_and_exposure']=True
    assert Autoencoder(True)(x).shape==x.shape
    fake=pd.DataFrame([dict(id='same',image_path='a',sha256='s',pixel_sha256='p',component='c',role=r) for r in ('source_fit','target_test')])
    try: check_separation(fake)
    except ValueError: checks['leakage_rejected']=True
    else: raise AssertionError('Leakage accepted')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'complete.json'; save_json(path,dict(fingerprint='test',status='complete',files=[]))
        assert completed(path,'test'); checks['resume_skip']=True
        folder=Path(directory)/'MetalParts-test1-conv-k0-s0'; folder.mkdir()
        save_json(folder/'complete.json',dict(fingerprint='test',status='complete',files=[]))
        fake_args=argparse.Namespace(out=directory,group='MetalParts',condition='test1',method='conv',k=0,seed=0,controlled=False,align=False)
        experiment(fake_args,{'fingerprint':'test'})
    y=np.array([0]*20+[1]*20); scores=np.arange(40,dtype=float)
    assert auc(y,scores)==roc_auc_score(y,scores)
    null=paired_inference([dict(y=y,a=scores[None],b=scores[None])]*6,199)
    strong=paired_inference([dict(y=y,a=scores[None],b=-scores[None])]*6,199)
    assert null['permutation_p']==1 and strong['permutation_p']<.05
    checks['auroc_reference_and_permutation_null_strong']=[null['permutation_p'],strong['permutation_p']]
    masks=np.zeros((2,5,5),bool); masks[0,0,0]=True; masks[1,2:4,2:4]=True
    maps=np.random.default_rng(2).integers(0,6,masks.shape).astype(float)
    components=[ndimage.label(m,structure=np.ones((3,3)))[0] for m in masks]
    points=[]
    for t in np.r_[np.inf,np.unique(maps)[::-1]]:
        pred=maps>=t; rates=[pred[i][c==r].mean() for i,c in enumerate(components) for r in range(1,c.max()+1)]
        points.append((pred[~masks].mean(),np.mean(rates)))
    xx,yy=np.array(points).T
    assert abs(aupro(masks,maps)-integrate_pro(xx,yy))<1e-7
    assert abs(aupro(masks,maps,both=True)['pixel_auroc']-roc_auc_score(masks.ravel(),maps.ravel()))<1e-12
    assert abs(aupro(masks,masks.astype(float))-1)<1e-7
    checks['aupro_brute_reference_ties_and_perfect']=True
    for cosine in (True,False):
        memory=Memory(cosine,{}); memory.fit([('a',x)],[],torch.device('cpu'))
        score,amap=memory.score(x); assert torch.isfinite(score).all() and amap.shape==(1,32,32)
        memory.fit([('a',x)],[('b',x*.9)],torch.device('cpu'),alignment=True)
        assert memory.info['alignment'] and torch.isfinite(memory.score(x)[0]).all()
    checks['memory_finite_and_grid']=True; checks['all_passed']=True
    save_json(ROOT/'comparison/verification/p2-selftest.json',checks); print(json.dumps(checks))
