"""CPU selftest; explicitly requested tiny local smoke; resumable individual study runs."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from common.data import ROOT, settings, save_json, digest, load_roles, subset, check_separation, Features, completed, selftest
from models.conv_ae.model import Autoencoder, optimize
from models.dinov2.model import Memory as DinoMemory
from models.patchcore.model import Memory as PatchCoreMemory
def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()
def clocked(function,device):
    sync(device); start=time.perf_counter(); value=function(); sync(device)
    return value,time.perf_counter()-start
def experiment(args,cfg):
    from common.analyze import final_metrics, recalibrated_metrics
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    name=f'{args.group}-{args.condition}-{args.method}-k{args.k}-s{args.seed}'+('-10plus10' if args.controlled else '')+('-mean' if args.align else '')
    folder=out/name; folder.mkdir(exist_ok=True); done=folder/'complete.json'
    if completed(done,cfg['fingerprint']): print('SKIP',name); return
    save_json(folder/'config.json',cfg)
    base_method='mlp' if args.method=='mlp' else ('conv' if args.method in ('conv','norm','no_replay') else args.method)
    base_args=copy.copy(args); base_args.method=base_method; base_args.k=0; base_args.controlled=False; base_args.align=False
    baseline=f'{args.group}-{args.condition}-{base_method}-k0-s{args.seed}'
    if args.k: experiment(base_args,cfg)
    start=time.perf_counter(); roles=load_roles(cfg,args.group,args.condition,args.k,args.seed,args.controlled)
    torch.manual_seed(args.seed); device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    backbone='cnn' if args.method=='patchcore' else 'dino'
    features=Features(cfg,backbone,device,args.cache)
    source_rows, val_rows, target_rows=(subset(roles,r) for r in ('source_fit','source_val','adaptation'))
    source=[features.get(r) for r in source_rows]; val=[features.get(r) for r in val_rows]
    target=[features.get(r) for r in target_rows]; features.close()
    cpdir=out/'checkpoints'; cpdir.mkdir(exist_ok=True)
    cp=cpdir/f'{args.group}-{args.condition}-{base_method}-s{args.seed if base_method in ("conv","mlp") else 0}.pt'
    fit_start=time.perf_counter(); info={}; is_ae=base_method in ('conv','mlp')
    if is_ae:
        detector=Autoencoder(base_method=='mlp').to(device)
        if cp.exists():
            record=torch.load(cp,map_location='cpu',weights_only=True)
            if record['fingerprint']!=cfg['fingerprint']: raise ValueError('Stale source checkpoint')
            detector.load_state_dict(record['state']); info=record['training']
        else:
            check_separation(roles)
            info=optimize(detector,source,val,[],cfg['source_steps'],1e-3,args.seed)
            record={'state':{k:v.cpu() for k,v in detector.state_dict().items()},'fingerprint':cfg['fingerprint'],
                    'feature_fingerprint':features.key,'training':info,'source_ids':[r['id'] for r in source_rows],'seed':args.seed}
            atomic_save(record,cp)
        threshold=(json.loads((out/baseline/'metrics.json').read_text())[0]['threshold'] if args.k else
                   float(np.quantile([detector.score(x.to(device))[0].item() for x in val],.95,method='higher')))
        if args.k:
            check_separation(roles)
            info=optimize(detector,source,val,target,cfg['adapt_steps'],1e-4,args.seed,
                          args.method if args.method in ('norm','no_replay') else 'decoder')
        info['parameters']=sum(p.numel() for p in detector.parameters())
        atomic_save({'state':{k:v.cpu() for k,v in detector.state_dict().items()},'fingerprint':cfg['fingerprint'],
                    'support_ids':[r['id'] for r in target_rows],'seed':args.seed,'training':info},folder/'detector.pt')
    else:
        detector=(DinoMemory if backbone=='dino' else PatchCoreMemory)(backbone=='dino',cfg); check_separation(roles)
        source_pairs=list(zip([r['id'] for r in source_rows],source))
        if cp.exists():
            record=torch.load(cp,map_location='cpu',weights_only=True)
            if record['fingerprint']!=cfg['fingerprint']: raise ValueError('Stale memory checkpoint')
            detector.bank=record['bank'].to(device); info=record['info']
        else:
            info=detector.fit(source_pairs,[],device)
            atomic_save({'bank':detector.bank.cpu(),'fingerprint':cfg['fingerprint'],'info':info},cp)
        threshold=(json.loads((out/baseline/'metrics.json').read_text())[0]['threshold'] if args.k else
                   float(np.quantile([detector.score(x.to(device))[0].item() for x in val],.95,method='higher')))
        if args.k:
            check_separation(roles); info=detector.fit(source_pairs,list(zip([r['id'] for r in target_rows],target)),device,args.align)
        info['deterministic_k0_alias']=args.k==0 and args.seed!=0
        atomic_save({'bank':detector.bank.cpu(),'fingerprint':cfg['fingerprint'],'info':info},folder/'detector.pt')
    sync(device); fit_seconds=time.perf_counter()-fit_start; predictions=[]; maps={}
    for domain in ('source_test','target_test','calibration'):
        if args.method=='mlp' and args.k==0 and domain!='source_test': continue
        for row in subset(roles,domain):
            x=features.get(row).to(device); score,amap=detector.score(x)
            predictions.append({'id':row['id'],'role':domain,'score':float(score.item())})
            if domain!='calibration': maps[(domain,row['id'])]=F.interpolate(amap[:,None],(448,448),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    features.close(); pd.DataFrame(predictions).to_csv(folder/'scores.csv',index=False)
    save_json(folder/'prediction-seal.json',{'score_hash':digest(folder/'scores.csv'),
        'map_hash':hashlib.sha256(b''.join(v.tobytes() for v in maps.values())).hexdigest(),'fingerprint':cfg['fingerprint']})
    for domain in ('source_test','target_test'):
        for key in [key for key in maps if key[0]==domain][:2]:
            np.save(folder/f'map-{key[0]}-{key[1]}.npy',maps[key][::4,::4],allow_pickle=False)
    metrics=final_metrics(cfg,args.group,args.condition,roles,predictions,maps,threshold)
    if args.controlled:
        calibrated=float(np.quantile([p['score'] for p in predictions if p['role']=='calibration'],.95,method='higher'))
        comparison=recalibrated_metrics(cfg,args.group,args.condition,predictions,metrics,calibrated)
        metrics+=comparison
    for row in metrics:
        row.update(info,group=args.group,method=args.method,condition=args.condition,k=args.k,seed=args.seed,alignment=args.align,
                   fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,fingerprint=cfg['fingerprint'])
        row.setdefault('threshold_policy','source-fixed threshold under score shift' if args.k else 'source-fixed K0')
    if args.k:
        base=json.loads((out/baseline/'metrics.json').read_text()); initial=next(r['image_auroc'] for r in base if r['domain']=='source_test')
        for row in metrics: row['source_retention']=metrics[0]['image_auroc']-initial
    if not args.k:
        for row in metrics: row['source_retention']=0.0
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    record={'group':args.group,'method':args.method,'condition':args.condition,'k':args.k,'seed':args.seed,'controlled':args.controlled,
            'alignment':args.align,'fingerprint':cfg['fingerprint'],'fit_seconds':fit_seconds,'parameters':info.get('parameters',0),
            'trainable_parameters':info['trainable_parameters'],'deterministic_k0_alias':info.get('deterministic_k0_alias',False),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0}
    for row in metrics:
        prefix=row['domain']+('_calibrated' if row['threshold_policy']=='10_adapt_10_calibration' else '')
        for key in ('image_auroc','image_auprc','pixel_auroc','aupro','f1','false_alarm_rate','recall','fpr_at_95_tpr','threshold','source_retention'):
            if key in row: record[prefix+'_'+key]=row[key]
    pd.DataFrame([record]).to_csv(raw/f'{name}.csv',index=False)
    save_json(folder/'metrics.json',metrics)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    save_json(folder/'runtime.json',{'elapsed_seconds':time.perf_counter()-start,'fit_seconds':fit_seconds,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0,
        'source_commit':commit.stdout.strip() if commit.returncode==0 else os.environ.get('DS_SOURCE_COMMIT','NOT FOUND (see code_hash)'),
        'code_hash':cfg['code_hash'],
        'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'feature_fingerprint':features.key})
    save_json(done,{'status':'complete','fingerprint':cfg['fingerprint'],'files':[
        {'path':p.name,'sha256':digest(p)} for p in folder.iterdir() if p.is_file() and p!=done]+
        [{'path':'../raw/'+name+'.csv','sha256':digest(raw/f'{name}.csv')}]})
    print('COMPLETE',name)
def smoke(args,cfg):
    from common.model import smoke_benchmark
    roles=load_roles(cfg,args.group,args.condition,5,0)
    rows=[subset(roles,r)[0] for r in ('source_fit','source_val','adaptation')]
    check_separation(roles); result=smoke_benchmark(cfg,rows,args.device)
    save_json(ROOT/'comparison/verification/p3-smoke.json',result); print(json.dumps(result))
def atomic_save(value,path):
    tmp=Path(str(path)+'.tmp'); torch.save(value,tmp); tmp.replace(path)
def schedule(queue,cfg):
    conditions=[f'test{i}' for i in range(1,7)]; core=['conv','dino','patchcore']; jobs=[]
    groups=[('MetalParts',conditions)] if queue!='Q4' else [('PCB',cfg['pcb_go_conditions'])]
    for group,domains in groups:
        if queue in ('Q1','Q4'):
            for k in (0,5,20):
                for c in domains:
                    for method in core:
                        for seed in range(5 if queue=='Q1' else 3): jobs.append((group,c,method,k,seed,False,False))
        else:
            methods=['no_replay','norm','mlp','dino','patchcore'] if queue=='Q2' else core
            for c in domains:
                for method in methods:
                    for seed in range(3): jobs.append((group,c,method,20,seed,queue=='Q3',queue=='Q2' and method in core))
    return jobs
def dispatch(args,cfg):
    import zipfile
    from common.model import timing_projection
    i,n=map(int,args.shard.split('/')); assert 0<=i<n
    base=Path(args.out); base.mkdir(parents=True,exist_ok=True)
    if args.timing:
        roles=load_roles(cfg,'MetalParts','test1',5,0)
        rows=[subset(roles,r)[0] for r in ('source_fit','source_val','adaptation')]
        from common.model import smoke_benchmark
        timing=smoke_benchmark(cfg,rows,args.device)
        table=timing_projection(timing,cfg); save_json(base/f'timing.shard{i}.json',table); print(json.dumps(table)); return
    if args.extract:
        for group in cfg['enabled_groups']:
            rows=pd.read_csv(ROOT/cfg['manifests']/group/f'{group}_roles.csv').drop_duplicates('id').sort_values('id').to_dict('records')[i::n]
            for backbone in ('dino','cnn'):
                features=Features(cfg,backbone,args.device,args.cache)
                for row in rows: features.get(row)
                features.close()
        print('EXTRACTION COMPLETE',i); return
    os.environ['CACHE_READ_ONLY']='1'
    jobs=schedule(args.queue,cfg); worker=base/f'shard{i}'; raw=base/'raw';raw.mkdir(exist_ok=True)
    for index,job in enumerate(jobs):
        if index%n!=i: continue
        if time.time()>float(os.environ.get('DEADLINE_EPOCH','inf')): raise TimeoutError('Approved wall-clock budget exhausted')
        a=copy.copy(args);a.group,a.condition,a.method,a.k,a.seed,a.controlled,a.align=job;a.out=str(worker)
        experiment(a,cfg)
        name=f'{a.group}-{a.condition}-{a.method}-k{a.k}-s{a.seed}'+('-10plus10' if a.controlled else '')+('-mean' if a.align else '')
        row=pd.read_csv(worker/'raw'/f'{name}.csv'); row['queue']=args.queue; row['job_index']=index
        path=raw/f'{a.method}.shard{i}.csv'; prior=pd.read_csv(path) if path.exists() else pd.DataFrame()
        merged=pd.concat([prior,row]).drop_duplicates(['queue','job_index'],keep='last');temp=path.with_suffix('.tmp');merged.to_csv(temp,index=False);temp.replace(path)
        lock=base/'.zip-lock'
        while True:
            try: lock.mkdir(); break
            except FileExistsError: time.sleep(.2)
        try:
            archive=base/f'results_{os.environ.get("QUEUE_BATCH",args.queue)}.zip'; temp=archive.with_suffix('.tmp')
            files=list(raw.glob('*.csv'))+list(base.glob('*.log'))+list(base.glob('timing*.json'))+list(base.glob('session*.json'))
            for w in base.glob('shard*'):
                files+=list((w/'checkpoints').glob('*.pt'))
                for done in w.glob('*/complete.json'): files+=list(done.parent.glob('*'))
                files+=list((w/'raw').glob('*.csv'))
            assert sum(p.stat().st_size for p in files)*2 < 15*1024**3,'Output/ZIP budget exceeded'
            with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
                for p in files:
                    if p.is_file() and p.suffix!='.tmp': z.write(p,p.relative_to(base))
            temp.replace(archive)
        finally: lock.rmdir()
    print('QUEUE COMPLETE',args.queue,'shard',i,'expected',len(jobs[i::n]))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'common/configs/config.yaml'))
    parser.add_argument('--selftest',action='store_true'); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--method',choices=['conv','dino','patchcore','norm','no_replay','mlp'],default='conv')
    parser.add_argument('--group',default='MetalParts'); parser.add_argument('--condition',default='test1')
    parser.add_argument('--k',type=int,choices=[0,5,20],default=0); parser.add_argument('--seed',type=int,choices=range(5),default=0)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--out',default=os.environ.get('OUT_DIR',str(ROOT/'comparison/run-output')))
    parser.add_argument('--queue',choices=['Q1','Q2','Q3','Q4']); parser.add_argument('--extract',action='store_true'); parser.add_argument('--timing',action='store_true'); parser.add_argument('--shard',default='0/1')
    parser.add_argument('--cache',default=os.environ.get('CACHE_DIR')); parser.add_argument('--controlled',action='store_true'); parser.add_argument('--align',action='store_true')
    args=parser.parse_args()
    if args.selftest: selftest(); return
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    cfg=settings(args.config); frozen=json.loads((ROOT/'docs/freeze.json').read_text())
    if cfg['fingerprint']!=frozen['fingerprint']: raise ValueError('Freeze mismatch: record decision, revalidate and refreeze')
    if args.queue or args.extract or args.timing: dispatch(args,cfg); return
    if args.group not in cfg['enabled_groups']: raise ValueError('Group NO-GO: approved manifests required')
    if (args.align or args.controlled or args.method in ('norm','no_replay')) and (args.k!=20 or args.seed>2):
        raise ValueError('Secondary arms require K=20, seeds 0..2')
    if args.align and args.method not in ('dino','patchcore'): raise ValueError('Alignment is memory-only')
    if args.align and args.controlled: raise ValueError('Alignment plus calibration is not a scheduled arm')
    if args.method=='mlp' and (args.k not in (0,20) or args.seed>2): raise ValueError('MLP: source K0 and K20, three seeds only')
    if args.controlled and args.method not in ('conv','dino','patchcore'): raise ValueError('Controlled table is core-only')
    if args.smoke: smoke(args,cfg)
    else: experiment(args,cfg)
if __name__=='__main__': main()
