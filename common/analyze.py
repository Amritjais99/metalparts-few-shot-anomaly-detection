"""Final label-consuming metrics and paired, condition-stratified inference."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from common.data import ROOT, save_json, digest

FALLBACK = 'conditional on image-level independence, potentially optimistic'


def auc(y, scores):
    y = np.asarray(y, dtype=bool); scores = np.asarray(scores)
    if not y.any() or y.all(): raise ValueError('AUROC requires both classes')
    ranks = rankdata(scores, axis=-1, method='average')
    positive, negative = y.sum(), (~y).sum()
    return (ranks[...,y].sum(-1)-positive*(positive+1)/2)/(positive*negative)


def image_metrics(y, score, threshold):
    y = np.asarray(y,int); score = np.asarray(score,float)
    if not np.isfinite(score).all(): raise ValueError('Nonfinite image scores')
    tn, fp, fn, tp = confusion_matrix(y,score>threshold,labels=[0,1]).ravel()
    fpr, tpr, _ = roc_curve(y,score,drop_intermediate=False)
    return {'image_auroc':float(roc_auc_score(y,score)), 'image_auprc':float(average_precision_score(y,score)),
            'f1':float(2*tp/max(1,2*tp+fp+fn)), 'false_alarm_rate':float(fp/(fp+tn)),
            'recall':float(tp/(tp+fn)), 'fpr_at_95_tpr':float(fpr[tpr>=.95].min()),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'normal_n':int(tn+fp),
            'anomaly_n':int(tp+fn),'threshold':float(threshold)}


def integrate_pro(x,y,limit=.3):
    stop = np.searchsorted(x,limit,side='right'); xx, yy = x[:stop],y[:stop]
    if xx[-1] < limit:
        v = yy[-1]+(y[stop]-yy[-1])*(limit-xx[-1])/(x[stop]-xx[-1])
        xx, yy = np.r_[xx,limit],np.r_[yy,v]
    return float(np.trapezoid(yy,xx)/limit)


def aupro(masks,maps,both=False):
    """Exact tied-score sweep, region-equal weights, 8-connectivity, all-image background.

    Definition: anomalib AUPRO docs (v1.0.0 metrics), FPR limit .30.
    Independent brute-threshold reference is checked in run.py --selftest.
    This is an equivalent definition, not a claim of bitwise library reproduction.
    """
    weights, regions = [], 0
    for mask in masks:
        components, count = ndimage.label(mask,structure=np.ones((3,3),int)); regions += count
        sizes = np.bincount(components.ravel()); sizes[0] = 1
        weights.append(np.where(mask,1/sizes[components],0).astype(np.float32))
    y = np.asarray(masks,bool).ravel(); scores = np.asarray(maps,np.float32).ravel()
    if not regions or y.all(): return {'aupro':None,'pixel_auroc':None} if both else None
    order = np.argsort(-scores,kind='stable'); sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),len(scores)-1]
    fpr = np.r_[0,np.cumsum(~y[order],dtype=np.float64)[ends]/(~y).sum()]
    pro = np.r_[0,np.cumsum(np.asarray(weights).ravel()[order],dtype=np.float64)[ends]/regions]
    result=integrate_pro(fpr,pro)
    if not both: return result
    tpr=np.r_[0,np.cumsum(y[order],dtype=np.float64)[ends]/y.sum()]
    return {'aupro':result,'pixel_auroc':integrate_pro(fpr,tpr,1.)}


def final_metrics(cfg,group,condition,roles,predictions,maps,threshold):
    """Call only AFTER predictions/seal are persisted. No adaptation code imports annotations."""
    annotation_path=ROOT/cfg['manifests']/group/f'{group}_evaluation_annotations.csv'
    if digest(annotation_path)!=cfg['evaluation_hashes'][group]: raise ValueError('Evaluation annotations changed')
    labels = pd.read_csv(annotation_path,keep_default_na=False)
    labels = labels[labels.condition == condition].set_index(['role','id'])
    rows = []
    for domain in ('source_test','target_test'):
        selected = [p for p in predictions if p['role']==domain]
        if not selected: continue
        masks, y = [], []
        for row in selected:
            annotation = labels.loc[(domain,row['id'])]; label = int(annotation.label); y.append(label)
            if label:
                if not annotation.mask_path: raise ValueError('Missing anomaly mask')
                with Image.open(ROOT/cfg['raw']/annotation.mask_path) as image:
                    masks.append(np.asarray(image.convert('L').resize((448,448),Image.Resampling.NEAREST))>0)
            else: masks.append(np.zeros((448,448),bool))
        predicted = np.stack([maps[(domain,p['id'])] for p in selected])
        result = image_metrics(y,[p['score'] for p in selected],threshold)
        result.update(domain=domain,**aupro(masks,predicted,both=True),aupro_implementation='exact_region_weighted_tied_sweep',
                      evaluation_pixels=448*448*len(y),
                      anomalies_without_positive_resized_mask=sum(label and not mask.any() for label,mask in zip(y,masks)))
        rows.append(result)
    return rows


def recalibrated_metrics(cfg,group,condition,predictions,metrics,threshold):
    labels=pd.read_csv(ROOT/cfg['manifests']/group/f'{group}_evaluation_annotations.csv')
    labels=labels[labels.condition==condition].set_index(['role','id']); rows=[]
    for original in metrics:
        selected=[p for p in predictions if p['role']==original['domain']]
        y=[int(labels.loc[(p['role'],p['id']),'label']) for p in selected]
        row=dict(original); row.update(image_metrics(y,[p['score'] for p in selected],threshold))
        row.update(threshold_policy='10_adapt_10_calibration',calibration_n=10); rows.append(row)
    return rows


def paired_inference(pairs, iterations=10000, seed=20261112, grouped=False):
    """Pairs contain y, a/b (seed x image), optional globally reviewed capture IDs.

    Seeds stay paired; AUROCs are averaged within condition then equally across conditions.
    Mixed-class groups stay whole and invalid bootstrap draws are rejected and counted.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean([np.mean(auc(p['y'],p['a'])-auc(p['y'],p['b'])) for p in pairs]))
    ranked = [(rankdata(p['a'],axis=-1)/len(p['y']),rankdata(p['b'],axis=-1)/len(p['y'])) for p in pairs]
    keys = [np.asarray(p['groups'] if grouped else [f'{i}:{j}' for j in range(len(p['y']))]) for i,p in enumerate(pairs)]
    unique = np.unique(np.concatenate(keys)); membership = {v:i for i,v in enumerate(unique)}
    indices = [np.array([membership[v] for v in k]) for k in keys]
    strata = {}
    for k in unique:
        labels = {int(y) for p,ks in zip(pairs,keys) for y in np.asarray(p['y'])[ks==k]}
        strata[k] = next(iter(labels)) if len(labels)==1 else 2
    strata_arrays = [np.array([membership[k] for k in unique if strata[k]==label]) for label in (0,1,2)]
    bootstrap, extreme, rejected, attempts = [], 0, 0, 0
    while len(bootstrap) < iterations:
        attempts += 1
        if attempts > iterations*100: raise ValueError('Too many class-empty grouped bootstrap draws')
        if grouped:
            sampled = np.concatenate([rng.choice(s,len(s),replace=True) for s in strata_arrays if len(s)])
            counts = np.bincount(sampled,minlength=len(unique))
            resampled = [np.repeat(np.arange(len(p['y'])),counts[ix]) for p,ix in zip(pairs,indices)]
        else:
            resampled = [np.concatenate([rng.choice(np.flatnonzero(np.asarray(p['y'])==v),
                           sum(np.asarray(p['y'])==v),replace=True) for v in (0,1)]) for p in pairs]
        if any(len(np.unique(np.asarray(p['y'])[ix]))<2 for p,ix in zip(pairs,resampled)):
            rejected += 1; continue
        bootstrap.append([np.mean([np.mean(auc(np.asarray(p['y'])[ix],np.asarray(p[method])[:,ix]))
                          for p,ix in zip(pairs,resampled)]) for method in ('a','b')])
    for _ in range(iterations):
        swaps = rng.integers(0,2,len(unique)).astype(bool); effects = []
        for p,(a,b),ix in zip(pairs,ranked,indices):
            mask = swaps[ix][None]
            effects.append(np.mean(auc(p['y'],np.where(mask,b,a))-auc(p['y'],np.where(mask,a,b))))
        extreme += abs(np.mean(effects)) >= abs(observed)-1e-12
    boot = np.asarray(bootstrap)
    return {'difference':observed,'ci95_difference':np.quantile(boot[:,0]-boot[:,1],[.025,.975]).tolist(),
            'ci95_a':np.quantile(boot[:,0],[.025,.975]).tolist(),'ci95_b':np.quantile(boot[:,1],[.025,.975]).tolist(),
            'permutation_p':(1+int(extreme))/(iterations+1),'resamples':iterations,
            'rejected_bootstrap':rejected,'assumption':'reviewed capture grouping' if grouped else FALLBACK}


def group_audit(cfg,group='MetalParts'):
    base = ROOT/cfg['manifests']/group; roles = pd.read_csv(base/f'{group}_roles.csv')
    annotations = pd.read_csv(base/f'{group}_evaluation_annotations.csv')
    merged = roles.merge(annotations,on=['condition','role','id'],validate='one_to_one')
    output = []
    for condition, rows in merged[merged.role=='target_test'].groupby('condition'):
        record = {'condition':condition,'reviewed_normal_groups':0,'reviewed_anomaly_groups':0}
        for label, name in ((0,'normal'),(1,'anomaly')):
            counts = rows[rows.label==label].component.value_counts()
            record[name+'_images'] = int(counts.sum()); record[name+'_candidate_components'] = len(counts)
            record[name+'_largest_fraction'] = float(counts.max()/counts.sum())
        record.update(bootstrap=FALLBACK,reason='Physical capture identities NOT FOUND; candidate hashes are not reviewed captures')
        output.append(record)
    pd.DataFrame(output).to_csv(ROOT/'comparison/verification/p2-group-counts.csv',index=False)
    save_json(ROOT/'comparison/verification/p2-group-counts.json',output)
    return output


def summarize(out,manifest_dir):
    """CPU P7 entry point. Reject incomplete paired grids; exactly one inferential comparison."""
    out=Path(out).resolve(); summaries=[]
    folders={}
    relocated = (out == ROOT/'comparison')
    ledgers = (list((ROOT/'models').glob('*/metrics/*.shard*.csv')) if relocated
               else list((out/'raw').glob('*.shard*.csv')))
    if ledgers:
        for ledger in ledgers:
            shard=ledger.stem.split('.shard')[-1]
            for r in pd.read_csv(ledger).to_dict('records'):
                name=f"{r['group']}-{r['condition']}-{r['method']}-k{int(r['k'])}-s{int(r['seed'])}"+('-10plus10' if r['controlled'] else '')+('-mean' if r['alignment'] else '')
                if name in folders: raise ValueError('Duplicate scheduled output')
                if relocated:
                    model_root = ledger.parents[1]
                    arm = {'no_replay':'no-replay', 'norm':'film-norm', 'mlp':'per-patch-mlp'}.get(r['method'])
                    if r['alignment']: arm = 'mean-alignment'
                    if arm: model_root = model_root/'ablations'/arm
                    folders[name] = model_root/'results'/f'shard{shard}'/name
                else:
                    folders[name]=out/f'shard{shard}'/name
    else: folders={p.name:p for p in out.iterdir() if (p/'complete.json').exists()}
    for folder in folders.values():
        if not (folder/'complete.json').exists(): raise ValueError('Missing completion record')
        for row in json.loads((folder/'metrics.json').read_text()):
            row['controlled']='10plus10' in folder.name; summaries.append(row)
    frame=pd.DataFrame(summaries)
    if frame.empty: raise ValueError('No completed results')
    keys=['group','method','condition','k','alignment','controlled','domain','threshold_policy']
    metrics=['image_auroc','image_auprc','pixel_auroc','aupro','false_alarm_rate','f1','fpr_at_95_tpr']
    frame.groupby(keys,dropna=False)[metrics].agg(['mean','std','count']).to_csv(out/('tables/summary.csv' if relocated else 'summary.csv'))
    labels=pd.read_csv(Path(manifest_dir)/'MetalParts_evaluation_annotations.csv')
    pairs=[]
    for condition in ('test1','test2','test3','test4','test5','test6'):
        annotation=labels[(labels.condition==condition)&(labels.role=='target_test')].set_index('id').sort_index()
        scores={}
        for method in ('conv','dino'):
            rows=[]
            for seed in range(5):
                folder=folders.get(f'MetalParts-{condition}-{method}-k20-s{seed}',out/'MISSING')
                if not (folder/'complete.json').exists(): raise ValueError(f'Incomplete primary grid: {folder}')
                table=pd.read_csv(folder/'scores.csv'); table=table[table.role=='target_test'].set_index('id').sort_index()
                if not table.index.equals(annotation.index): raise ValueError('Unpaired test IDs')
                rows.append(table.score.to_numpy())
            scores[method]=np.stack(rows)
        pairs.append(dict(y=annotation.label.to_numpy(),a=scores['conv'],b=scores['dino']))
    inference=paired_inference(pairs)
    save_json(out/('statistics/primary-inference.json' if relocated else 'primary-inference.json'),inference)
    print(json.dumps(inference))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    parser.add_argument('--manifests',default=str(ROOT/'data/manifests/MetalParts'))
    args=parser.parse_args(); summarize(args.out,args.manifests)
