"""CPU-only metadata, integrity, duplicate and fixed-membership audit; no model code."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv
import zipfile
import hashlib
import json
from pathlib import Path
import random
import time
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/raw'
OUT = ROOT/'comparison/verification/p1-audit'
REV = '82c642b2d18be60f2dab104f8dd4dea1f4ccfacb'

def csvout(name, rows, fields=None):
    rows = list(rows)
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def readcsv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def readjsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def stable_id(group,split,name):
    return hashlib.sha256(f'{group}\0{split}\0{name}'.encode()).hexdigest()[:24]

def safe(rel):
    path=(RAW/rel).resolve()
    assert path.is_relative_to(RAW.resolve()), rel
    return path

def hashrow(row):
    path=safe(row['image_path'])
    with path.open('rb') as f:
        row['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    with Image.open(path) as im:
        rgb=im.convert('RGB'); w,h=rgb.size
        row.update(width=w,height=h,pixel_sha256=hashlib.sha256(
            f'{w}x{h}'.encode()+b'\0'+rgb.tobytes()).hexdigest(),
            phash=str(imagehash.phash(rgb,hash_size=8,highfreq_factor=4)))
    row['mask_exists']=bool(row['mask_path'] and safe(row['mask_path']).is_file())
    row['mask_geometry_ok']=True
    row['mask_binary']=True
    if row['mask_exists']:
        with Image.open(safe(row['mask_path'])) as im:
            row['mask_geometry_ok']=im.size==(w,h)
            values=set(np.unique(np.asarray(im.convert('L'))).tolist())
            row['mask_binary']=values.issubset({0,1,255})
    return row

class UF:
    def __init__(self,ids): self.p={i:i for i in ids}
    def find(self,i):
        while self.p[i]!=i:
            self.p[i]=self.p[self.p[i]]; i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.p[max(a,b)]=min(a,b)

def pairs_for(rows):
    values=np.array([int(r['phash'],16) for r in rows],dtype=np.uint64)
    pairs=[]
    for start in range(0,len(rows),128):
        distances=np.bitwise_count(values[start:start+128,None]^values[None,:]).astype(np.uint8)
        ii,jj=np.where(distances<=4)
        for i,j in zip(ii.tolist(),jj.tolist()):
            i+=start
            if i>=j: continue
            a,b=rows[i],rows[j]
            pairs.append(dict(group=a['group'],id_a=a['id'],id_b=b['id'],
                hamming=int(distances[i-start,j]),exact_bytes=a['sha256']==b['sha256'],
                exact_rgb=a['pixel_sha256']==b['pixel_sha256'],
                split_a=a['split'],split_b=b['split']))
    return pairs

def table(rows,keys):
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+
        ['| '+' | '.join(str(r.get(k,'')) for k in keys)+' |' for r in rows])

def main():
    started=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    available='--hash-available' in sys.argv
    partial='--hash-metalparts' in sys.argv or available
    verification=None if partial else json.loads((ROOT/'common/configs/p1-download-verification.json').read_text())
    if not partial:
        assert verification['revision']==REV and not verification['errors']
    inventory_path=ROOT/'data/manifests/shared/image_inventory.csv'
    if inventory_path.exists():
        rows=readcsv(inventory_path)
        for r in rows:
            for k in ['label','width','height']: r[k]=int(r[k])
            for k in ['mask_exists','mask_geometry_ok','mask_binary']: r[k]=r[k]=='True'
    else:
        rows=[]
    cached={r['id']:r for r in rows}; pending=[]; expected=[]
    for group in (['MetalParts'] if partial and not available else ['MetalParts','PCB']):
        for meta in sorted((RAW/group).glob('*/metadata.jsonl')):
            split=meta.parent.name.rsplit('_',1)[-1]
            for entry in readjsonl(meta):
                assert entry['label'] in (0,1)
                name=entry['file_name'].replace('\\','/')
                if available and (not (meta.parent/name).is_file() or
                    (entry.get('mask') and not safe(entry['mask']).is_file())):
                    continue
                rid=stable_id(group,split,name); expected.append(rid)
                if rid not in cached:
                    pending.append(dict(id=rid,group=group,split=split,
                        image_path=(meta.parent/name).relative_to(RAW).as_posix(),
                        file_name=name,label=entry['label'],mask_path=entry.get('mask') or '',
                        metadata_file=meta.relative_to(RAW).as_posix()))
    assert len(set(expected))==len(expected)
    if pending:
        with ThreadPoolExecutor(max_workers=4) as pool:
            hashed=[]
            for i,row in enumerate(pool.map(hashrow,pending)):
                hashed.append(row)
                if i%100==0: print(f'Audit hashes {i+1}/{len(pending)}',flush=True)
        cached.update({r['id']:r for r in hashed})
    rows=[cached[rid] for rid in expected]; csvout('image_inventory.csv',rows)
    if partial:
        print(f'Available audit hashes saved: {len(rows)}',flush=True); return
    byid={r['id']:r for r in rows}
    allpairs=[]; clusters=[]
    for group in ['MetalParts','PCB']:
        group_rows=[r for r in rows if r['group']==group]
        pairs=pairs_for(group_rows); allpairs+=pairs
        uf=UF(r['id'] for r in group_rows)
        for p in pairs: uf.join(p['id_a'],p['id_b'])
        members=defaultdict(list)
        for r in group_rows:
            r['candidate_component']=uf.find(r['id']); members[uf.find(r['id'])].append(r)
        for cid,items in members.items():
            if len(items)>1:
                clusters.append(dict(group=group,component=cid,images=len(items),
                    test_images=sum(r['split'].startswith('test') for r in items),
                    splits=','.join(sorted({r['split'] for r in items})),
                    members=';'.join(r['id'] for r in items),identity='UNVERIFIED pHash candidate component'))
    csvout('near_duplicate_pairs.csv',allpairs)
    cross_group=[p for p in pairs_for(rows) if byid[p['id_a']]['group']!=byid[p['id_b']]['group']]
    csvout('cross_group_candidates.csv',cross_group,
           fields=['group','id_a','id_b','hamming','exact_bytes','exact_rgb','split_a','split_b'])
    csvout('near_duplicate_clusters.csv',clusters)
    csvout('image_inventory.csv',rows)
    raw_counts=[]
    for group,split in sorted({(r['group'],r['split']) for r in rows}):
        rr=[r for r in rows if (r['group'],r['split'])==(group,split)]
        raw_counts.append(dict(group=group,split=split,normal=sum(r['label']==0 for r in rr),
            anomalous=sum(r['label']==1 for r in rr),masks=sum(r['mask_exists'] for r in rr),
            missing_anomaly_masks=sum(r['label']==1 and not r['mask_exists'] for r in rr),
            bad_geometry=sum(not r['mask_geometry_ok'] for r in rr),
            nonbinary_masks=sum(not r['mask_binary'] for r in rr),
            resolutions=','.join(sorted({f"{r['width']}x{r['height']}" for r in rr}))))
    csvout('raw_split_counts.csv',raw_counts)
    shifts={'MetalParts':{'train':'source','test0':'source','test1':'lighting','test2':'position',
            'test3':'rotation','test4':'scale','test5':'background 1','test6':'background 2'},
            'PCB':{'train':'source Set1','test0':'source Set2','test1':'lighting',
            'test2':'white balancing','test3':'rotation','test4':'position','test5':'shadow'}}
    structure=[]
    for count in raw_counts:
        sample=next(r for r in rows if r['group']==count['group'] and r['split']==count['split'] and r['label']==0)
        structure.append(dict(group=count['group'],split=count['split'],
            shift_inferred_from_filename=shifts[count['group']][count['split']],
            normal_filename_example=sample['file_name'],object_id='NOT DOCUMENTED'))
    csvout('domain_structure.csv',structure)
    # Existing held-out memberships are immutable. Build fresh normal-only draws,
    # preserving seeds 0..2 exactly, extending only within the old support pool.
    counts=[]; assignments=[]; draws=[]; annotations=[]; problems=[]; pcb=[]; crossrole=[]
    historical=zipfile.Path(ROOT/'comparison/archive/pilot-224px.zip', at='data/manifests/expanded/MetalParts/')
    for condition in [f'test{i}' for i in range(1,7)]:
        old=readjsonl(historical/condition/'manifest.jsonl')
        chosen={r['id']:r['role'] for r in old}
        assert all(r['id'] in byid and byid[r['id']]['image_path']==r['image_path']
                   and byid[r['id']]['sha256']==r['sha256'] for r in old)
        active=[byid[r['id']] for r in old]
        relevant=[r for r in rows if r['group']=='MetalParts' and r['split'] in ['train','test0',condition]]
        uf=UF(r['id'] for r in relevant); valid_ids=set(uf.p)
        pp=[p for p in allpairs if p['id_a'] in valid_ids and p['id_b'] in valid_ids]
        for p in pp: uf.join(p['id_a'],p['id_b'])
        role_sets=defaultdict(set)
        for r in active: role_sets[uf.find(r['id'])].add(chosen[r['id']])
        mixed={c:roles for c,roles in role_sets.items() if len(roles)>1}
        forbidden={c:roles for c,roles in mixed.items() if not roles <= {'source_test','target_test'}}
        for cid,roles in mixed.items(): crossrole.append(dict(group='MetalParts',condition=condition,
            component=cid,roles=','.join(sorted(roles)),classification='FIT/CALIBRATION COLLISION' if cid in forbidden else 'evaluation-only dependence candidate'))
        for r in active:
            role=chosen[r['id']]
            if role in ['source_fit','source_val','target_support_pool']: assert r['label']==0
            assignments.append(dict(condition=condition,role=role,id=r['id'],image_path=r['image_path'],
                sha256=r['sha256'],component=uf.find(r['id']),group='MetalParts'))
            if role in ['source_val','source_test','target_test']:
                annotations.append(dict(condition=condition,role=role,id=r['id'],label=r['label'],mask_path=r['mask_path']))
        pool=sorted([r for r in old if r['role']=='target_support_pool'],key=lambda r:r['component_id'])
        previous={(d['seed'],d['k']):d['ids'] for d in readjsonl(historical/condition/'support_draws.jsonl')}
        failed=[]
        for seed in range(5):
            order=[r['id'] for r in pool]; random.Random(seed).shuffle(order)
            if seed<3:
                assert order[:20]==previous[(seed,20)]
            for k in [0,5,20]:
                for rank,rid in enumerate(order[:k]):
                    draws.append(dict(condition=condition,seed=seed,k=k,arm='all_k_adaptation',role='adaptation',rank=rank,id=rid))
            selected=order[:20]
            if len(selected)!=20 or len({uf.find(i) for i in selected})!=20: failed.append(seed)
            for rank,rid in enumerate(selected):
                draws.append(dict(condition=condition,seed=seed,k=20,arm='controlled_10_plus_10',
                    role='adaptation' if rank<10 else 'calibration',rank=rank,id=rid))
        count=lambda role,label=None:sum(chosen[r['id']]==role and (label is None or r['label']==label) for r in active)
        c=dict(condition=condition,source_fit=count('source_fit'),source_val=count('source_val'),
            support=count('target_support_pool'),target_normal=count('target_test',0),target_anomalous=count('target_test',1),
            support_components=len({uf.find(r['id']) for r in pool}),cross_role_components=len(mixed),
            forbidden_role_components=len(forbidden),
            k20_and_10plus10='PASS' if not failed and not forbidden else 'BLOCKED',failed_seeds=','.join(map(str,failed)))
        raw_target=[r for r in relevant if r['split']==condition]
        c['excluded_target_normals']=sum(r['label']==0 for r in raw_target)-c['support']-c['target_normal']
        c['excluded_target_anomalies']=sum(r['label']==1 for r in raw_target)-c['target_anomalous']
        counts.append(c)
        if forbidden or failed: problems.append(condition)
    csvout('MetalParts_roles.csv',assignments)
    csvout('MetalParts_support_calibration_draws.csv',draws)
    csvout('MetalParts_evaluation_annotations.csv',annotations)
    csvout('MetalParts_counts.csv',counts)
    csvout('MetalParts_run_schedule.csv',
        [dict(condition=c['condition'],seed=seed,k=k,adaptation_images=k,
              calibration_images=0,arm='all_k_adaptation',status=c['k20_and_10plus10'])
         for c in counts for seed in range(5) for k in [0,5,20]] +
        [dict(condition=c['condition'],seed=seed,k=20,adaptation_images=10,
              calibration_images=10,arm='controlled_10_plus_10',status=c['k20_and_10plus10'])
         for c in counts for seed in range(3)])
    csvout('cross_role_candidates.csv',crossrole,fields=['group','condition','component','roles','classification'])
    old_counts={r['condition']:r for r in readcsv(zipfile.Path(ROOT/'comparison/archive/pilot-224px.zip', at='docs/tables/data_splits.csv'))}
    differences=[]
    for c in counts:
        old=old_counts[c['condition']]
        for a,b in [('source_fit','source_fit'),('source_val','source_val'),('support','target_support_pool'),
                    ('target_normal','target_test_normal'),('target_anomalous','target_test_anomaly')]:
            if c[a]!=int(old[b]): differences.append(dict(condition=c['condition'],field=a,old=old[b],new=c[a]))
    csvout('old_count_differences.csv',differences,fields=['condition','field','old','new'])
    # PCB feasibility uses conservative candidates, never relaxes pHash or claims
    # that giant similarity components establish repeated physical captures.
    for condition in [f'test{i}' for i in range(1,6)]:
        rr=[r for r in rows if r['group']=='PCB' and r['split'] in ['train','test0',condition]]
        uf=UF(r['id'] for r in rr); ids=set(uf.p)
        pp=[p for p in allpairs if p['id_a'] in ids and p['id_b'] in ids]
        for p in pp: uf.join(p['id_a'],p['id_b'])
        source_test={uf.find(r['id']) for r in rr if r['split']=='test0'}
        target_all={uf.find(r['id']) for r in rr if r['split']==condition}
        fit={uf.find(r['id']) for r in rr if r['split']=='train' and r['label']==0}-source_test-target_all
        eligible={uf.find(r['id']) for r in rr if r['split']==condition and r['label']==0}-source_test
        target=[r for r in rr if r['split']==condition]
        normal=sum(r['label']==0 for r in target); anomalous=sum(r['label']==1 for r in target)
        mask_ok=all(r['mask_exists'] and r['mask_geometry_ok'] and r['mask_binary'] for r in target if r['label']==1)
        # Hold back >=20 normal components before reserving support.
        capacity=min(40,max(0,len(eligible)-20))
        go=len(fit)>=30 and capacity>=20 and anomalous>0 and mask_ok
        component_sizes=Counter(uf.find(r['id']) for r in rr)
        touched={i for p in pp for i in [p['id_a'],p['id_b']]}
        pcb.append(dict(condition=condition,raw_target_normal=normal,raw_target_anomalous=anomalous,
            anomaly_masks=sum(r['label']==1 and r['mask_exists'] for r in target),mask_ok=mask_ok,
            source_fit_plus_val_components=len(fit),eligible_target_normal_components=len(eligible),
            support_capacity_after_test_reserve=capacity,reviewed_independent_support='NOT ESTABLISHED',
            near_duplicate_pairs=len(pp),candidate_clusters=sum(n>1 for n in component_sizes.values()),
            target_images_affected=sum(r['id'] in touched for r in target),
            status='REVIEW REQUIRED' if go else 'NO-GO',
            reason='Unreviewed pHash candidates cannot establish capture independence' if go else 'Conservative source/support component capacity fails; physical identity unverified'))
    csvout('PCB_go_no_go.csv',pcb)
    summary=dict(revision=REV,images=len(rows),raw_counts=raw_counts,metalparts=counts,
        differences=differences,pcb=pcb,pcb_decision='GO' if any(p['status']=='GO' for p in pcb) else 'NO-GO',metalparts_blocked=problems,
        near_duplicate_pairs=len(allpairs),candidate_clusters=len(clusters),
        cross_group_candidate_pairs=len(cross_group),
        affected_official_test_images=len({i for p in allpairs for i in [p['id_a'],p['id_b']] if byid[i]['split'].startswith('test')}),
        exact_byte_pairs=sum(p['exact_bytes'] for p in allpairs),exact_rgb_pairs=sum(p['exact_rgb'] for p in allpairs),
        elapsed_seconds=time.time()-started,seeds=list(range(5)),k=[0,5,20],
        selection='Preserved historical roles/IDs. Normal-only pool; deterministic Python random shuffle; old seeds verified.',
        object_identity='NOT ESTABLISHED: metadata has file_name,label,mask, no documented physical ID')
    (OUT/'audit-summary.json').write_text(json.dumps(summary,indent=2))
    content='# P1 data audit — 2026-11-13\n\n'
    content+=table([dict(item='Dataset',value=f'AmazonScience/RobustAD @ {REV}'),
        dict(item='License',value='CC BY 4.0; pinned README.md, common/configs/p1-dataset-listing.json'),
        dict(item='Integrity',value=f"{verification['verified_files']} listed group files verified by repository hash; README separate"),
        dict(item='Splits',value='Official train/test0 and MetalParts test1..6, PCB test1..5. Few-shot roles are custom, not official support splits.'),
        dict(item='Image access',value='CPU decoding for audit hashes/resolution/masks only; no model inference/training/GPU'),
        dict(item='Evidence',value='data/manifests/{MetalParts,PCB,shared}/*.csv; metadata paths retained in image_inventory.csv')],['item','value'])
    content+='\n\n## Official raw split counts\n\n'+table(raw_counts,list(raw_counts[0]))
    content+='\n\n## Domains and file structure\n\n'+table(structure,list(structure[0]))
    content+='\n\n| Review | Limitation |\n|---|---|\n| visual_review.csv | Three samples visually inspected; PCB layout similarity is not evidence of physical-object identity. Domain names above are filename-based interpretations, not undocumented identity annotations. |\n'
    content+='\n\n## Preserved MetalParts experiment counts\n\n'+table(counts,list(counts[0]))
    content+='\n\n| Comparison | Finding |\n|---|---|\n| Old data_splits.csv | '+('No differences in source-fit, source-val, support or held-out class counts.' if not differences else json.dumps(differences))+' |\n'
    content+='| Raw vs experiment counts | Raw official test folders include reserved supports and excluded duplicates; raw counts should not equal held-out test counts. |\n'
    content+='| Sampling | Seeds 0..2 draws match historical files; seeds 3..4 extend the same normal pool. K=5 nested in K=20; controlled split first 10 adapt, last 10 calibrate. K=0 has no support rows. |\n'
    content+='| Threshold pool | source_val role in MetalParts_roles.csv; never source replay. Five independent source-training seeds are future P2/P4 work. |\n'
    content+='| Annotation barrier | Training roles CSV and draws have no defect labels/masks; final labels in separate evaluation_annotations.csv. Audit inventory is not a training manifest. |\n'
    content+='\n## Duplicate/capture audit\n\n'+table([dict(check=k,value=summary[k]) for k in ['near_duplicate_pairs','candidate_clusters','affected_official_test_images','exact_byte_pairs','exact_rgb_pairs','cross_group_candidate_pairs']],['check','value'])
    content+='\n\n| Evidence | Interpretation |\n|---|---|\n| near_duplicate_pairs.csv / near_duplicate_clusters.csv | Exhaustive within-group all-split pHash <=4 candidates and component member IDs; exact flags separate. Test affected means official test0..N, not necessarily experiment leakage. |\n| cross_role_candidates.csv | Fit/calibration collisions block a condition. Links solely between source-test and target-test are evaluation-dependence warnings, retained for grouped uncertainty review; they do not expose test data to fitting. |\n| Filenames | MetalParts names encode normal/defect text, shift and an integer; PCB names encode capture settings/index and sometimes Set1/Set2. No documented physical-object ID; do not parse defect words for sampling. |\n| Capture identity | Similarity/filename IDs do not prove physical independence; global components identify possible dependence across condition comparisons. |\n'
    content+='\n## PCB confirmation gate\n\n'+table(pcb,list(pcb[0]))
    content+='\n\n| Decision | Consequence |\n|---|---|\n| NO-GO | No PCB model run. Report MetalParts development-only evidence; no unseen-domain confirmation or broad industrial generalization claim. Giant pHash clusters need independently justified capture review, not a relaxed threshold chosen to pass. |\n'
    content+='\n## Reproduce and status\n\n| Item | Value |\n|---|---|\n| Download | Existing Python: `python common/p1_download.py` (resumable; pinned listing and disk gate) |\n| Audit | `python common/p1_audit.py` (CPU only; cached image hash inventory reused if present) |\n| Hash rerun | Protected inventory is reused; audit output goes to comparison/verification/p1-audit. |\n| Next | Stop before P2. Any blocked MetalParts condition needs explicit resolution before training. |\n'
    (OUT/'data-audit.md').write_text(content,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['images','metalparts','pcb_decision','differences','near_duplicate_pairs','candidate_clusters','elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
