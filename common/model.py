"""Frozen-feature reconstruction and explicitly budgeted memory variants."""
import numpy as np
import torch
from models.conv_ae.model import Autoencoder, optimize
from models.dinov2.model import Memory as DinoMemory
from models.patchcore.model import Memory as PatchCoreMemory


def smoke_benchmark(cfg,rows,device):
    """Three authorized normals only; two source/adaptation updates; never loads annotations."""
    import time
    from common.data import ROOT, Features
    if device!='cuda': raise ValueError('P3 smoke requires CUDA')
    if any(r['role'] not in ('source_fit','source_val','adaptation') for r in rows):
        raise ValueError('Smoke cannot access held-out data')
    start=time.perf_counter(); output={'scope':'source-fit/source-val/one support normal ONLY',
        'image_ids':[r['id'] for r in rows], 'device':torch.cuda.get_device_name(0),
        'torch':torch.__version__, 'test_metrics_computed':False, 'fingerprint':cfg['fingerprint']}
    def timed(fn):
        torch.cuda.synchronize(); t=time.perf_counter(); value=fn(); torch.cuda.synchronize()
        return value,time.perf_counter()-t
    def deadline(_=None):
        if time.perf_counter()-start>900: raise TimeoutError('15-minute smoke budget exhausted')
    dino=None
    for backbone in ('dino','cnn'):
        deadline(); torch.cuda.reset_peak_memory_stats()
        features=Features(cfg,backbone,'cuda',__import__('os').environ.get('CACHE_DIR',str(ROOT/'cache/smoke')))
        if list(features.folder.glob('*.npy')): raise ValueError('Timing requires a fresh temporary timing cache')
        _,load_seconds=timed(features.load); tensors,seconds=timed(lambda:[features.get(r) for r in rows])
        output[backbone]={'extraction_seconds':seconds,'images_per_second':len(rows)/seconds,
            'initialization_seconds':load_seconds,'shape':list(tensors[0].shape),
            'cache_bytes_per_image':sum(p.stat().st_size for p in features.folder.glob('*.npy'))/len(rows),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        features.close(); memory=(DinoMemory if backbone=='dino' else PatchCoreMemory)(backbone=='dino',cfg)
        # Repeated authorized features benchmark the full candidate/bank sizes, not study fitting.
        _,fit_seconds=timed(lambda:memory.fit([(f'timing-{i}',tensors[0]) for i in range(64)],[],torch.device('cuda')))
        _,score_seconds=timed(lambda:memory.score(tensors[1].to('cuda')))
        output[backbone].update(memory_fit_8192_candidates_seconds=fit_seconds,
            knn_image_seconds=score_seconds,knn_bank=2048,timing_pool='64 repetitions of one authorized source feature grid')
        del memory; torch.cuda.empty_cache()
        if backbone=='dino': dino=tensors
    for mlp,name in ((False,'conv'),(True,'mlp')):
        deadline(); torch.manual_seed(0); model=Autoencoder(mlp).cuda(); torch.cuda.reset_peak_memory_stats()
        _,train_seconds=timed(lambda:optimize(model,[dino[0]],[dino[1]],[],2,1e-3,0,callback=deadline))
        scores,score_seconds=timed(lambda:model.score(dino[1].cuda()))
        if not torch.isfinite(scores[0]).all(): raise ValueError('Nonfinite smoke reconstruction')
        record={'source_two_steps_seconds':train_seconds,'score_image_seconds':score_seconds,
                'parameters':sum(p.numel() for p in model.parameters())}
        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for mode in (('decoder',) if mlp else ('decoder','norm','no_replay')):
            model.load_state_dict(original); check,seconds=timed(lambda:optimize(model,[dino[0]],[dino[1]],[dino[2]],2,1e-4,0,mode,deadline))
            record[mode+'_two_steps_seconds']=seconds; record[mode+'_trainable']=check['trainable_parameters']
        record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        output[name]=record; del model; torch.cuda.empty_cache()
    output['elapsed_seconds']=time.perf_counter()-start
    return output


def timing_projection(t,cfg):
    from common.data import ROOT
    import pandas as pd
    r=pd.read_csv(ROOT/cfg['manifests']/'MetalParts'/'MetalParts_roles.csv'); unique=r.id.nunique()
    total=sum(((r.condition==c)&r.role.isin(['source_test','target_test'])).sum() for c in r.condition.unique())
    extract=unique*sum(t[b]['extraction_seconds']/3 for b in ('dino','cnn'))/2
    source=30*1000*t['conv']['source_two_steps_seconds']/2
    score=15*total*sum(t[b]['knn_image_seconds'] for b in ('dino','cnn'))
    # CPU pixel timing is measured here on synthetic arrays, never held-out maps.
    import time
    from common.analyze import aupro
    rng=np.random.default_rng(23); maps=rng.random((24,448,448),dtype=np.float32)
    masks=np.zeros(maps.shape,bool); masks[12:,50:100,40:90]=True
    start=time.perf_counter();aupro(masks,maps,both=True);cpu=(time.perf_counter()-start)/24
    q1=extract+(source+score+45*total*cpu*1.3+60*100*t['conv']['decoder_two_steps_seconds']/2)/2
    q2=(18*1000*t['mlp']['source_two_steps_seconds']/2+3*total*sum(t[b]['knn_image_seconds'] for b in ('dino','cnn'))+15*total*cpu*1.3)/2
    q3=(3*total*sum(t[b]['knn_image_seconds'] for b in ('dino','cnn'))+9*total*cpu*1.3)/2
    hours={k:float(v*1.5/3600) for k,v in dict(Q1=q1,Q2=q2,Q3=q3,Q4=0,session2=extract+q2+q3).items()}
    return dict(queue_hours_estimate_two_shards=hours,measurements=t,cpu_synthetic_seconds_per_image=cpu,
                note='50% margin includes bank construction, validation, checkpoint loads, two-process CPU contention; add actual setup/preflight time. Not a guarantee.',
                projected_two_sessions_hours=hours['Q1']+hours['session2'],test_scores_accessed=False)
