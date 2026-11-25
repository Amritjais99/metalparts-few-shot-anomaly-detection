"""Report the frozen MetalParts experiments from saved outputs (CPU only).

Anomalous is positive. Decisions are score > saved threshold. For Q1/Q2 use
the source-normal 95th quantile: source-fixed threshold under score shift.
Q3 compares the two saved thresholds on the identical 10-adapted detector.
Counts are computed within seed and condition (TP/FN/FP/TN), then averaged
over seeds. Summed counts add condition counts per seed before aggregation.
Rates are per-run ratios; undefined precision is unavailable. Accuracy can
hide class imbalance. Balanced accuracy (TPR + TNR)/2 and AUROC are headline
metrics. AUPRC is average precision; FPR@95%TPR is the minimum achieved FPR
at TPR >= .95, without interpolation. Pixel AUROC and AUPRO are saved values
with mask checks, never derived from the small display maps. Macro means
weight conditions equally. SD is the sample SD of the seed-level values.
"""
import argparse
from ast import literal_eval
from bisect import bisect_right
from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
import random
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXED = "source-fixed threshold under score shift"
RECAL = "target-recalibrated"
FOOT = "MetalParts development set, image-level independence assumed"
TESTS = [f"test{i}" for i in range(1, 7)]
METHODS = {
    "conv": ("Conv AE", "#b44a42", "conv_ae"),
    "dino": ("DINOv2", "#226a95", "dinov2"),
    "patchcore": ("PatchCore", "#237759", "patchcore"),
    "no_replay": ("Conv AE · no replay", "#b44a42", "conv_ae"),
    "norm": ("Conv AE · FiLM / norm-only", "#b44a42", "conv_ae"),
    "mlp": ("Per-patch MLP", "#b44a42", "conv_ae"),
    "dino_mean": ("DINOv2 · mean alignment", "#226a95", "dinov2"),
    "patchcore_mean": ("PatchCore · mean alignment", "#237759", "patchcore"),
}
MAIN = list(METHODS)[:3]
ARMS = {"no_replay": "no-replay", "norm": "film-norm", "mlp": "per-patch-mlp"}
OUTCOMES = {"TP": "Correct defect", "FP": "False alarm", "FN": "Missed defect", "TN": "Correct normal"}
COUNTS = ["tp", "fn", "fp", "tn"]
DENOM = ["normal_n", "anomaly_n"]
RATES = ["recall", "specificity", "fpr", "fnr", "precision", "accuracy", "balanced_accuracy", "f1"]
RANKING = ["image_auroc", "image_auprc", "fpr_at_95_tpr", "pixel_auroc", "aupro"]
RETENTION = ["source_auroc", "source_k0_auroc", "source_retention"]
GRID = [i / 200 for i in range(201)]
KEYS = ["variant", "k", "policy"]
EXPECTED = {"conv": [.650, .666, .666], "dino": [.846, .886, .906], "patchcore": [.829, .834, .869]}


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def close(x, y):
    if pd.isna(x) and pd.isna(y):
        return True
    return math.isclose(x, y, rel_tol=1e-12, abs_tol=2e-13)


def ratio(n, d):
    return n / d if d else math.nan


def binary_counts(labels, scores, threshold):
    truth, pred = labels.eq(1), scores.gt(threshold)
    tp, fn, fp, tn = (int(v.sum()) for v in (truth & pred, truth & ~pred, ~truth & pred, ~truth & ~pred))
    p, n = tp + fn, tn + fp
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, normal_n=n, anomaly_n=p,
                recall=tp/p, specificity=tn/n, fpr=fp/n, fnr=fn/p, precision=ratio(tp,tp+fp),
                accuracy=(tp+tn)/(p+n), balanced_accuracy=(tp/p+tn/n)/2, f1=2*tp/(2*tp+fp+fn))


def ranking(labels, scores):
    n_pos = int(labels.sum())
    n_neg = len(labels)-n_pos
    if not n_pos or not n_neg:
        raise ValueError("Both test classes must be present")
    auc = (scores.rank()[labels.eq(1)].sum()-n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    tie = labels.groupby(scores).agg(["sum", "size"]).sort_index(ascending=False).cumsum()
    tpr = tie["sum"]/n_pos
    fpr = (tie["size"]-tie["sum"])/n_neg
    ap = ((tie["sum"]/tie["size"])*tpr.diff().fillna(tpr)).sum()
    return dict(image_auroc=auc, image_auprc=ap, fpr_at_95_tpr=fpr[tpr.ge(.95)].min()), fpr, tpr


def roc_grid(fpr, tpr):
    x, y = [0.,*fpr], [0.,*tpr]
    for point in GRID:
        left = bisect_right(x,point)-1
        if x[left] == point:
            yield point,y[left]
        else:
            right = left+1
            yield point,y[left]+(y[right]-y[left])*(point-x[left])/(x[right]-x[left])


def evidence():
    rows, seen = [], set()
    for ledger in sorted((ROOT/"models").glob("*/metrics/*.shard*.csv")):
        for r in pd.read_csv(ledger, float_precision="round_trip").to_dict("records"):
            queue, method = r["queue"], r["method"]
            key = (queue, r["job_index"])
            if key in seen:
                raise ValueError(f"Duplicate scheduled result {key}")
            seen.add(key)
            controlled, aligned = bool(r["controlled"]), bool(r["alignment"])
            variant = method + "_mean" if aligned else method
            arm = "mean-alignment" if aligned else ARMS.get(method)
            root = ledger.parents[1]
            if arm:
                root = root/"ablations"/arm
            name = f"MetalParts-{r['condition']}-{method}-k{r['k']}-s{r['seed']}"
            name += "-10plus10" if controlled else "-mean" if aligned else ""
            folder = root/"results"/f"shard{ledger.stem[-1]}"/name
            if not (folder/"complete.json").is_file():
                raise ValueError(f"Incomplete result {folder}")
            rows.append(dict(r, variant=variant, folder=folder))
    assert Counter(r["queue"] for r in rows) == {"Q1":270, "Q2":90, "Q3":54}
    return rows


def labels_and_inputs(manifests):
    paths = pd.read_csv(manifests/"MetalParts_roles.csv").set_index(["condition", "role", "id"]).sort_index()
    annotation_path = manifests/"MetalParts_evaluation_annotations.csv"
    cfg = json.loads((ROOT/"common/configs/config.yaml").read_text(encoding="utf-8"))
    assert sha(annotation_path) == cfg["evaluation_hashes"]["MetalParts"]
    labels = pd.read_csv(annotation_path, keep_default_na=False).set_index(["condition", "role", "id"]).sort_index()
    return labels, paths


def read_scores(row, labels, roles=("source_test", "target_test")):
    p = row["folder"]/"scores.csv"
    seal = json.loads(p.with_name("prediction-seal.json").read_text(encoding="utf-8"))
    assert sha(p) == seal["score_hash"] and seal["fingerprint"] == row["fingerprint"]
    scores = pd.read_csv(p, float_precision="round_trip")
    tests = {}
    for role in roles:
        test = scores[scores.role.eq(role)].set_index("id").sort_index().score
        annotation = labels.loc[(row["condition"], role)]
        assert test.index.is_unique and test.index.equals(annotation.index) and test.map(math.isfinite).all()
        tests[role] = test.to_frame().join(annotation)
    return tests


def headline_gate(rows, labels):
    values = []
    for r in rows:
        if r["queue"] == "Q1":
            t = read_scores(r, labels)["target_test"]
            values.append((r["method"], r["k"], r["condition"], ranking(t.label, t.score)[0]["image_auroc"]))
    table = pd.DataFrame(values, columns=["method", "k", "condition", "auroc"])
    macro = table.groupby(["method","k","condition"]).auroc.mean().groupby(["method","k"]).mean()
    old = pd.read_csv(ROOT/"comparison/tables/q1-macro.csv").set_index(["method","k"]).image_auroc
    for method, expected in EXPECTED.items():
        for k, value in zip((0,5,20), expected):
            if round(macro[method,k],3) != value or not close(macro[method,k],old[method,k]):
                raise ValueError(f"STOP: AUROC mismatch for {method}, K={k}: {macro[method,k]}")
    print("PASS headline AUROC gate: all nine values match.", flush=True)


def measure(rows, labels):
    values, roc_rows, scores, source0 = [], [], {}, {}
    for r in rows:
        info = {k:r[k] for k in ("method","variant","queue","condition","k","seed","folder","fingerprint")}
        tests = read_scores(r, labels)
        t, s = tests["target_test"], tests["source_test"]
        ranking_values, fpr, tpr = ranking(t.label, t.score)
        source_auc = ranking(s.label, s.score)[0]["image_auroc"]
        policies = [(FIXED, "target_test_")]
        if r["queue"] == "Q3":
            policies += [(RECAL, "target_test_calibrated_")]
        base_method = "conv" if r["method"] in ("norm","no_replay") else r["method"]
        if r["k"] == 0:
            source_k0 = source_auc
        else:
            root = ROOT/"models"/METHODS[r["variant"]][2]
            if base_method == "mlp":
                root = root/"ablations"/ARMS["mlp"]
            name = f"MetalParts-{r['condition']}-{base_method}-k0-s{r['seed']}"
            shard = r["folder"].parent.name
            base = root/"results"/shard/name
            if not base.is_dir():
                base = root/"results"/("shard0" if shard == "shard1" else "shard1")/name
            if base not in source0:
                src = read_scores(dict(r, folder=base), labels, ("source_test",))["source_test"]
                prior = json.loads((base/"metrics.json").read_text(encoding="utf-8"))
                threshold0 = next(rec["threshold"] for rec in prior if rec["domain"] == "source_test")
                source0[base] = ranking(src.label,src.score)[0]["image_auroc"], threshold0
            source_k0, threshold0 = source0[base]
            assert close(threshold0, r["target_test_threshold"]), r["folder"]
        assert close(source_auc-source_k0, r["target_test_source_retention"])
        saved = json.loads((r["folder"]/"metrics.json").read_text(encoding="utf-8"))
        for policy,prefix in policies:
            recorded = next(v for v in saved if v["domain"] == "target_test" and
                            (v["threshold_policy"] == "10_adapt_10_calibration") == (policy == RECAL))
            threshold = recorded["threshold"]
            assert close(threshold,r[prefix+"threshold"])
            metrics = binary_counts(t.label,t.score,threshold) | ranking_values
            for key in ("image_auroc","image_auprc","fpr_at_95_tpr","recall","f1"):
                assert close(metrics[key],r[prefix+key]),(r["folder"],policy,key)
            assert close(metrics["fpr"],r[prefix+"false_alarm_rate"])
            assert all(metrics[key] == recorded[key] for key in COUNTS+DENOM)
            valid_masks = recorded.get("evaluation_pixels",0) > 0 and recorded.get("anomalies_without_positive_resized_mask") == 0
            pixel = {m:recorded.get(m) if valid_masks else None for m in ("pixel_auroc","aupro")}
            assert all(close(v,r[prefix+m]) for m,v in pixel.items() if v is not None)
            pixel_note = "Saved full-mask evaluation" if all(pd.notna(v) for v in pixel.values()) else "not available: valid pixel metrics absent in saved run"
            values.append(dict(info,**metrics,**pixel,policy=policy,threshold=threshold,pixel_status=pixel_note,
                               source_n=len(s),source_auroc=source_auc,source_k0_auroc=source_k0,source_retention=source_auc-source_k0))
        if r["queue"] != "Q3":
            roc_rows.extend(dict(info,fpr=x,tpr=y) for x,y in roc_grid(fpr,tpr))
            if r["k"] == 20:
                scores[r["variant"],r["condition"],r["seed"]] = (t,r["folder"],r["target_test_threshold"])
    return pd.DataFrame(values), pd.DataFrame(roc_rows), scores


def aggregate(per_run, keys, fields):
    g = per_run.groupby(keys,sort=False,dropna=False)
    result = g[fields].agg(["mean","std","count"])
    result.columns = [f"{metric}_{stat}" for metric,stat in result.columns]
    return result.join(g.seed.nunique().rename("seeds")).reset_index()


def mean_counts(per_run, keys):
    table = aggregate(per_run,keys,COUNTS+DENOM)
    for field, denom in {"tp":"anomaly_n","fn":"anomaly_n","fp":"normal_n","tn":"normal_n"}.items():
        table[field+"_row_pct"] = table[field+"_mean"]/table[denom+"_mean"]*100
    return table


class Outputs:
    def __init__(self, comparison):
        self.comparison = comparison
        self.paths = []
        self.missing = Counter()

    def csv(self, folder, stem, table):
        path = folder/f"{stem}.csv"
        path.parent.mkdir(parents=True,exist_ok=True)
        table.to_csv(path,index=False,na_rep="not available")
        self.paths.append(path)

    def plot(self, folder, stem, fig, title, threshold=False):
        path = folder/f"{stem}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.suptitle(title,fontsize=13)
        fig.text(.5,.012,FOOT+(", "+FIXED if threshold else ""),ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.94))
        fig.savefig(path,dpi=200,facecolor="white")
        plt.close(fig)
        self.paths.append(path)


def styling():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.grid":True,"grid.alpha":.2,
                         "axes.axisbelow":True,"savefig.dpi":200,"figure.facecolor":"white"})


def context(table):
    if table.empty:
        raise ValueError("Missing figure data")
    used = table["seeds"].unique() if "seeds" in table else [table.seed.nunique()]
    ns = "/".join(str(int(n)) for n in sorted(used,reverse=True))
    if "condition" in table and "normal_n_mean" in table:
        n = table.drop_duplicates("condition").eval("normal_n_mean + anomaly_n_mean").sum()
    else:
        col = "normal_n_mean" if "normal_n_mean" in table else "normal_n"
        an = "anomaly_n_mean" if col.endswith("_mean") else "anomaly_n"
        n = table.iloc[0][col] + table.iloc[0][an]
    return f"{ns} seeds · {int(n)} target images"


def confusion(ax, row, title):
    counts = [[row.tn_mean,row.fp_mean],[row.fn_mean,row.tp_mean]]
    pct = [[row.tn_row_pct,row.fp_row_pct],[row.fn_row_pct,row.tp_row_pct]]
    cmap = LinearSegmentedColormap.from_list(row.variant, ["#f5f6f8", METHODS[row.variant][1]])
    ax.imshow(pct,vmin=0,vmax=100,cmap=cmap)
    for y in range(2):
        for x in range(2):
            ax.text(x,y,f"{counts[y][x]:.1f}\n{pct[y][x]:.1f}%",ha="center",va="center",
                    color="white" if pct[y][x]>73 else "#152538")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Normal","Anomalous"],yticklabels=["Normal","Anomalous"],
           xlabel="Predicted class",ylabel="Actual class",title=title)
    ax.grid(False)


def roc_axes(ax):
    ax.plot([0,1],[0,1],"--",color=".65",linewidth=1)
    ax.set(xlabel="False positive rate",ylabel="True positive rate",xlim=(0,1),ylim=(0,1.01))
    ax.legend(loc="lower right",frameon=False)


def hist_table(selected, rank_normalise=False):
    # Histogram probabilities average per-run class fractions. No seed pooling of raw scores.
    histograms = []
    for variant,condition,seed,t,threshold in selected:
        v = t.score.rank(pct=True) if rank_normalise else t.score
        histograms.append((variant,condition,seed,t.label,v,threshold))
    if not histograms:
        return pd.DataFrame()
    bounds = [(v.min(),v.max()) for _,_,_,_,v,_ in histograms]
    lo,hi = (0.,1.) if rank_normalise else (min(x[0] for x in bounds),max(x[1] for x in bounds))
    if hi == lo:
        hi = lo + 1e-12
    step = (hi-lo)/20
    rows = []
    for m,c,seed,y,v,tau in histograms:
        bins = ((v-lo)/step).astype(int).clip(0,19)
        for label in (0,1):
            counts = bins[y.eq(label)].value_counts()
            for i in range(20):
                rows.append(dict(variant=m,condition=c,seed=seed,label=label,
                                 low=lo+i*step,high=lo+(i+1)*step,share=counts.get(i,0)/y.eq(label).sum(),threshold=tau))
    table = pd.DataFrame(rows)
    group = ["variant","label","low","high"]
    if not rank_normalise:
        group.append("condition")
    table["mean_share"] = table.groupby(group).share.transform("mean")
    return table


def hist_step(ax, table, label, color, linestyle, legend):
    x = table[table.label.eq(label)].drop_duplicates(["low","high"]).sort_values("low")
    ax.stairs(x.mean_share,[*x.low,x.high.iloc[-1]],label=legend,color=color,linestyle=linestyle,linewidth=1.6)


def per_model(outputs, per_run, condition, macro, conf, roc, scores, paths):
    for variant, (label,color,model) in METHODS.items():
        directory = ROOT/"models"/model/"metrics"
        tdir,fdir = directory/"tables",directory/"figures"
        c = condition[condition.variant.eq(variant)]
        m = macro[macro.variant.eq(variant)].sort_values("k")
        q = conf[conf.variant.eq(variant)]
        r = roc[roc.variant.eq(variant)]
        seeds,n = int(m.seeds.iloc[0]),int(m.normal_n_mean.iloc[0]+m.anomaly_n_mean.iloc[0])
        annotation=f"{seeds} seeds · {n} target images"
        ks=m.k.tolist()
        extra = c[KEYS+["condition","seeds"]+[f"{x}_{s}" for x in RETENTION for s in ("mean","std")]]
        outputs.csv(tdir,variant+"-per-condition",c)
        outputs.csv(tdir,variant+"-macro",m)
        outputs.csv(tdir,variant+"-confusion",q)
        outputs.csv(tdir,variant+"-source-retention",pd.concat([extra,m[extra.columns.difference(["condition"])].assign(condition="macro")],ignore_index=True))
        summed_per_seed = per_run[per_run.variant.eq(variant)].groupby(KEYS+["seed"],as_index=False)[COUNTS+DENOM].sum().assign(condition="sum")
        outputs.csv(tdir,variant+"-by-seed",pd.concat([per_run[per_run.variant.eq(variant)].drop(columns="folder"),summed_per_seed],ignore_index=True))
        outputs.csv(tdir,variant+"-roc",r)
        fig,ax=plt.subplots(figsize=(7.6,4.5))
        for condition_name,g in c.groupby("condition"):
            g=g.sort_values("k")
            ax.errorbar(g.k,g.image_auroc_mean,yerr=g.image_auroc_std,fmt="o-",color=color,alpha=.38,
                        linewidth=1,markersize=3,label=condition_name)
        ax.errorbar(m.k,m.image_auroc_mean,yerr=m.image_auroc_std,fmt="o-",color=color,linewidth=3,label="Macro mean")
        ax.set(xlabel="Target normal support images (K)",ylabel="Image AUROC",xticks=ks,ylim=(0,1.04))
        ax.legend(ncol=2,frameon=False)
        outputs.plot(fdir,variant+"-auroc",fig,f"{label} | Image AUROC (mean ± seed SD)\nK={','.join(map(str,ks))} · {annotation}")
        summed=q[q.condition.eq("sum")].sort_values("k")
        fig,axes=plt.subplots(1,len(summed),squeeze=False,figsize=(4.0*len(summed),4))
        for ax,row in zip(axes.flat,summed.itertuples()):
            confusion(ax,row,f"K={row.k} · N={int(row.normal_n_mean)} / A={int(row.anomaly_n_mean)}")
        outputs.plot(fdir,variant+"-confusion",fig,f"{label} | Summed mean counts and row percentages\n{annotation}",True)
        fig,axes=plt.subplots(2,3,figsize=(10,7))
        for ax,row in zip(axes.flat,q[q.condition.ne("sum")&q.k.eq(20)].sort_values("condition").itertuples()):
            confusion(ax,row,f"{row.condition} · N={int(row.normal_n_mean)} / A={int(row.anomaly_n_mean)}")
        outputs.plot(fdir,variant+"-confusion-k20-by-condition",fig,f"{label} | K=20 confusion by condition\n{annotation}",True)
        fig,ax=plt.subplots(figsize=(6.8,4.5))
        for (k,g),style in zip(r.groupby("k"),["--","-.", "-"] if len(ks)>1 else ["-"]):
            ax.plot(g.fpr,g.tpr,color=color,linestyle=style,label=f"K={k}")
        roc_axes(ax);outputs.plot(fdir,variant+"-roc",fig,f"{label} | Mean ROC on a common FPR grid\nK={','.join(map(str,ks))} · {annotation}")
        fig,axes=plt.subplots(2,3,figsize=(11.4,6.4)); h=[]
        for ax,cond in zip(axes.flat,TESTS):
            entries=[(m,c,s,*item[::2]) for (m,c,s),item in scores.items() if m==variant and c==cond]
            hist=hist_table(entries);h.append(hist)
            hist_step(ax,hist,0,color,"--","Normal");hist_step(ax,hist,1,color,"-","Anomalous")
            for tau in hist.threshold.unique():
                ax.axvline(tau,color="black",alpha=.22,linewidth=.85)
            ax.set_xlim(min(hist.low.min(),hist.threshold.min()),max(hist.high.max(),hist.threshold.max()))
            ax.set(title=f"{cond} · n={len(entries[0][3])}",xlabel="Raw anomaly score",ylabel="Fraction per bin")
        axes.flat[0].legend(frameon=False,fontsize=8)
        outputs.csv(tdir,variant+"-score-histograms-k20",pd.concat(h,ignore_index=True))
        outputs.plot(fdir,variant+"-scores-k20",fig,f"{label} | K=20 normal and anomalous scores\n{annotation} · Black lines: saved thresholds by seed",True)
        qualitative(outputs,fdir,variant,[variant],scores,paths,annotation)


def read_map(path):
    # The frozen exporter saves C-contiguous, 2-D little-endian float32 NPY.
    with path.open("rb") as f:
        if f.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError(f"Unexpected NPY version: {path}")
        header = literal_eval(f.read(struct.unpack("<H",f.read(2))[0]).decode("latin1"))
        data = f.read()
    h,w = header["shape"]
    assert header["descr"] == "<f4" and not header["fortran_order"] and len(data) == h*w*4
    return list(struct.iter_unpack(f"<{w}f",data))


def classify(row,tau):
    return ("T" if bool(row.label)==bool(row.score>tau) else "F") + ("P" if row.score>tau else "N")


def find_images(scores, condition, variant, outcome):
    for (v,c,seed),(table,folder,tau) in scores.items():
        if (v,c)!=(variant,condition):
            continue
        for amap in folder.glob("map-target_test-*.npy"):
            identity = amap.stem.removeprefix("map-target_test-")
            item = table.loc[identity]
            if classify(item,tau)==outcome:
                yield identity, seed, item, amap, tau


def qualitative(out, destination, stem, variants, scores, paths, annotation):
    entries = []
    for outcome,desc in OUTCOMES.items():
        width = 2+len(variants)
        # Comparison uses Conv AE outcomes and displays the other methods' verdicts.
        choices = [min(find_images(scores,c,variants[0],outcome),key=lambda x:x[:2],default=None) for c in TESTS]
        fig,axes = plt.subplots(6,width,figsize=(2*width,10.2 if any(choices) else 4.2),squeeze=False)
        for i,(condition,choice) in enumerate(zip(TESTS,choices)):
            for ax in axes[i]:
                ax.set_axis_off()
            if choice is None:
                axes[i,0].text(.03,.5,f"{condition} · {outcome} absent from saved map subset",ha="left",va="center",fontsize=8)
                out.missing[stem]+=1
                entries.append(dict(panel=stem,condition=condition,outcome=outcome,status="no saved map for outcome"))
                continue
            identity,seed,row,amap,tau = choice
            source = ROOT/"data/raw"/paths.loc[(condition,"target_test",identity),"image_path"]
            try:
                picture = plt.imread(source)
            except (OSError,ValueError):
                axes[i,0].text(.02,.5,f"{condition}\nOriginal unavailable",ha="left")
                out.missing[stem]+=1
                entries.append(dict(panel=stem,condition=condition,outcome=outcome,seed=seed,
                                    image_id=identity,status="original image unavailable"))
                continue
            extent = (0,picture.shape[1],picture.shape[0],0)
            step = max(1,max(picture.shape[:2])//600)
            picture = picture[::step,::step]
            mask = ROOT/"data/raw"/row.mask_path
            axes[i,0].imshow(picture,extent=extent)
            axes[i,0].set_title(f"{condition} · seed {seed}\n{identity[:10]}",fontsize=8)
            if row.label==0:
                axes[i,1].imshow([[0,0],[0,0]],vmin=0,vmax=1,cmap="gray")
                axes[i,1].set_title("GT mask · normal",fontsize=8)
            elif mask.is_file():
                axes[i,1].imshow(plt.imread(mask)[::step,::step],cmap="gray")
                axes[i,1].set_title("GT mask",fontsize=8)
            else:
                axes[i,1].text(.5,.5,"GT mask\nunavailable",ha="center",va="center")
                out.missing[stem+":mask"]+=1
            for j,variant in enumerate(variants,2):
                table,folder,t = scores[variant,condition,seed]
                model_row = table.loc[identity]
                match = folder/f"map-target_test-{identity}.npy"
                label = METHODS[variant][0]
                verdict = classify(model_row,t)
                if match.is_file():
                    axes[i,j].imshow(picture,extent=extent)
                    axes[i,j].imshow(read_map(match),extent=extent,alpha=.42,cmap="inferno",interpolation="bilinear")
                    status = "saved map"
                else:
                    axes[i,j].text(.5,.5,"Map absent\nfor same image",ha="center",va="center")
                    out.missing[stem+":comparison-map"]+=1
                    status = "no saved map for same image"
                axes[i,j].set_title(f"{label} · {verdict}",fontsize=8,color=METHODS[variant][1])
                entries.append(dict(panel=stem,condition=condition,outcome=outcome,variant=variant,seed=seed,
                                    image_id=identity,actual=int(row.label),predicted_outcome=verdict,
                                    score=float(model_row.score),threshold=t,
                                    original=source.relative_to(ROOT).as_posix(),mask=row.mask_path,
                                    map_file=match.relative_to(ROOT).as_posix() if match.is_file() else "",status=status))
        lab = " · selected by Conv AE verdict" if len(variants)>1 else ""
        label = "Comparison" if len(variants)>1 else METHODS[variants[0]][0]
        out.plot(destination,stem+f"-examples-{outcome.lower()}",fig,
                 f"{label} | K=20 {desc} examples{lab}\n{annotation}",True)
    tdir = destination.parent/"tables"
    out.csv(tdir,stem+"-example-selection",pd.DataFrame(entries))


def figure_comparison(out, per_run, condition, macro, confusion_table, roc, scores, paths):
    tdir, fdir = out.comparison/"tables",out.comparison/"figures"
    c = condition[condition.variant.isin(MAIN)]
    m = macro[macro.variant.isin(MAIN)]
    sums = confusion_table[confusion_table.condition.eq("sum")]
    annotation = context(m)
    fig,ax=plt.subplots(figsize=(8,4.7))
    for i,method in enumerate(MAIN):
        v=m[m.variant.eq(method)].sort_values("k")
        ax.bar([j+(i-1)*.24 for j in range(3)],v.image_auroc_mean,yerr=v.image_auroc_std,width=.23,
               color=METHODS[method][1],label=METHODS[method][0],capsize=3)
    ax.set(xticks=[0,1,2],xticklabels=[0,5,20],xlabel="Target normal support images (K)",ylabel="Image AUROC",ylim=(0,1.05))
    ax.legend(frameon=False);out.plot(fdir,"report-macro-auroc-bars",fig,f"Image AUROC by method (mean ± seed SD)\nK=0/5/20 · {annotation}")
    fig,ax=plt.subplots(figsize=(8,4.7))
    for method in MAIN:
        color = METHODS[method][1]
        for _,g in c[c.variant.eq(method)].groupby("condition"):
            g=g.sort_values("k")
            ax.plot(g.k,g.image_auroc_mean,".-",color=color,alpha=.2)
        v=m[m.variant.eq(method)].sort_values("k")
        ax.plot(v.k,v.image_auroc_mean,"o-",color=color,label=METHODS[method][0],linewidth=2.5)
    ax.set(xlabel="Target normal support images (K)",ylabel="Image AUROC",xticks=[0,5,20],ylim=(0,1))
    ax.legend(frameon=False);out.plot(fdir,"report-support-gain",fig,f"Image AUROC vs support size · thin lines show conditions\nK=0/5/20 · {annotation}")
    for k in (0,20):
        arr = c[c.k.eq(k)].pivot(index="variant",columns="condition",values="image_auroc_mean").loc[MAIN,TESTS]
        fig,ax=plt.subplots(figsize=(8,3.4)); im=ax.imshow(arr,vmin=0,vmax=1,cmap="YlGnBu")
        for i in range(len(arr)):
            for j in range(len(arr.columns)):
                v=arr.iloc[i,j];ax.text(j,i,f"{v:.3f}",ha="center",va="center",color="white" if v>.68 else "#17263c")
        ax.set(xticks=range(6),xticklabels=TESTS,yticks=range(3),yticklabels=[METHODS[x][0] for x in MAIN])
        for tick,method in zip(ax.get_yticklabels(),MAIN):
            tick.set_color(METHODS[method][1])
        ax.grid(False);fig.colorbar(im,ax=ax,label="Image AUROC")
        out.plot(fdir,f"report-auroc-heatmap-k{k}",fig,f"Image AUROC by condition | K={k}\n{annotation}")
        fig,ax=plt.subplots(figsize=(7,4.7))
        for method in MAIN:
            r=roc[roc.variant.eq(method)&roc.k.eq(k)]
            ax.plot(r.fpr,r.tpr,color=METHODS[method][1],label=METHODS[method][0])
        roc_axes(ax);out.plot(fdir,f"report-roc-k{k}",fig,f"Mean ROC · common FPR grid | K={k}\n{annotation}")
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.25))
    for method,ax in zip(MAIN,axes):
        confusion(ax,sums[sums.variant.eq(method)&sums.k.eq(20)].iloc[0],METHODS[method][0])
    out.plot(fdir,"report-confusion-k20",fig,f"K=20 confusion · summed mean counts and row percentages\n{annotation}",True)
    fig,ax=plt.subplots(figsize=(8,4.8))
    histogram=hist_table([(v,c,s,*t[::2]) for (v,c,s),t in scores.items() if v in MAIN],rank_normalise=True)
    out.csv(tdir,"report-ranked-score-histograms-k20",histogram)
    for method in MAIN:
        h=histogram[histogram.variant.eq(method)]
        for label,style,cat in [(0,"--","normal"),(1,"-","anomalous")]:
            hist_step(ax,h,label,METHODS[method][1],style,f"{METHODS[method][0]} · {cat}")
    ax.set(xlabel="Score percentile within condition and seed",ylabel="Mean class fraction per bin",xlim=(0,1))
    ax.legend(ncol=2,frameon=False,fontsize=8)
    out.plot(fdir,"report-ranked-scores-k20",fig,f"Rank-normalised K=20 anomaly scores\n{annotation}")
    fig,ax=plt.subplots(figsize=(8,4.5))
    for method in MAIN:
        v=m[m.variant.eq(method)].sort_values("k")
        ax.errorbar(v.k,v.source_retention_mean,yerr=v.source_retention_std,color=METHODS[method][1],
                    fmt="o-",label=METHODS[method][0],capsize=3)
    source_n=int(per_run.query("queue == 'Q1' and variant == 'conv' and k == 0 and seed == 0").source_n.sum())
    ax.axhline(0,color=".5",linewidth=1)
    ax.set(xlabel="Target normal support images (K)",ylabel="Source AUROC change from K=0",xticks=[0,5,20])
    ax.legend(frameon=False);out.plot(fdir,"report-source-retention",fig,
                                     f"Source retention (mean ± seed SD)\nK=0/5/20 · 5 seeds · {source_n} source image entries across conditions")
    qualitative(out,fdir,"comparison",MAIN,scores,paths,annotation)
    return q23_and_efficiency(out,per_run,source_n)


def q23_and_efficiency(out, rows, source_n):
    tdir,fdir=out.comparison/"tables",out.comparison/"figures"
    target_n=int(rows.query("queue == 'Q1' and variant == 'conv' and k == 20 and seed == 0").eval("normal_n+anomaly_n").sum())
    q2=rows[rows.queue.eq("Q2")| (rows.queue.eq("Q1")&rows.k.eq(20)&rows.seed.lt(3))]
    q2 = q2.groupby(["variant","seed"]).image_auroc.mean().groupby(level=0).agg(["mean","std"]).loc[list(METHODS)]
    out.csv(tdir,"report-q2-auroc",q2.reset_index())
    fig,ax=plt.subplots(figsize=(9,5.1))
    ax.barh(range(len(q2)),q2["mean"],xerr=q2["std"],color=[METHODS[x][1] for x in q2.index],capsize=3)
    ax.set(yticks=range(len(q2)),yticklabels=[METHODS[x][0] for x in q2.index],xlabel="Image AUROC",xlim=(0,1.04))
    ax.invert_yaxis();out.plot(fdir,"report-q2-ablations",fig,f"Q2 ablations and matched baselines (mean ± seed SD)\nK=20 · 3 seeds · {target_n} target images")
    q3=rows[rows.queue.eq("Q3")]
    q3macro=aggregate(q3.groupby(["variant","policy","seed"],as_index=False)[RATES+RANKING].mean(),["variant","policy"],RATES+RANKING)
    q3counts=mean_counts(q3.groupby(["variant","policy","seed"],as_index=False)[COUNTS+DENOM].sum(),["variant","policy"])
    out.csv(tdir,"report-q3-macro",q3macro);out.csv(tdir,"report-q3-confusion",q3counts)
    out.csv(tdir,"report-q3-by-seed",q3.drop(columns="folder"))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for metric,label,ax in zip(("fpr","recall"),("False-alarm rate","Defect recall"),axes):
        for i,method in enumerate(MAIN):
            for j,policy in enumerate((FIXED,RECAL)):
                r=q3macro[q3macro.variant.eq(method)&q3macro.policy.eq(policy)].iloc[0]
                ax.bar(i+(j-.5)*.32,r[metric+"_mean"],yerr=r[metric+"_std"],width=.31,
                       color=METHODS[method][1],alpha=.47 if j==0 else 1,hatch="//" if j==0 else "",
                       label=("Source-fixed" if j==0 else "Target-recalibrated") if i==0 else None,capsize=2)
        ax.set(xticks=range(3),xticklabels=[METHODS[m][0] for m in MAIN],ylabel=label,ylim=(0,1))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=8)
    out.plot(fdir,"report-q3-calibration",fig,f"Q3 calibration · same 10-adapted detectors (mean ± seed SD)\nK=20 (10+10) · 3 seeds · {target_n} target images",True)
    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    for i,policy in enumerate((FIXED,RECAL)):
        for j,method in enumerate(MAIN):
            r=q3counts[q3counts.variant.eq(method)&q3counts.policy.eq(policy)].iloc[0]
            confusion(axes[i,j],r,METHODS[method][0]+(" · source-fixed" if i==0 else " · recalibrated"))
    out.plot(fdir,"report-q3-confusion",fig,f"Q3 confusion · summed mean counts and row percentages\nK=20 (10+10) · 3 seeds · {target_n} target images",True)

    inference=json.loads((ROOT/"comparison/statistics/primary-inference.json").read_text(encoding="utf-8"))
    est=inference["difference"]
    lo,hi=inference["ci95_difference"]
    out.csv(tdir,"report-k20-paired-effects",pd.DataFrame([dict(
        contrast="Conv AE minus DINOv2",k=20,estimate=est,ci95_low=lo,ci95_high=hi,
        predeclared_permutation_p=inference["permutation_p"],assumption=inference["assumption"])]))
    fig,ax=plt.subplots(figsize=(8.8,3.1))
    ax.errorbar(est,0,xerr=[[est-lo],[hi-est]],fmt="o",color=METHODS["conv"][1],capsize=5)
    ax.axvline(0,color=".55",linestyle="--");ax.set(yticks=[0],yticklabels=["Conv AE − DINOv2"],xlabel="Paired difference in image AUROC",xlim=(-.36,.045),ylim=(-.65,.65))
    ax.text(est,-.31,f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]",ha="center")
    out.plot(fdir,"report-k20-paired-effects",fig,
             f"K=20 paired bootstrap 95% CI · predeclared permutation p={inference['permutation_p']:.6f}\n5 seeds · {target_n} target images")
    efficiency(out,rows,source_n)
    return inference,q2,q3macro


def efficiency(out,rows,source_n):
    runtime=pd.read_json(ROOT/"comparison/tables/runtime.json")
    runtime=runtime.set_index("run_path",verify_integrity=True)
    records=[]
    timings=[json.loads(p.read_text(encoding="utf-8"))["measurements"]
             for p in sorted((ROOT/"comparison/archive/q1-q3").glob("timing.shard*.json"))]
    all_rows=rows[rows.policy.eq(FIXED)&~rows.queue.eq("Q3")]
    for variant,(name,_,_) in METHODS.items():
        selected=all_rows[all_rows.variant.eq(variant)&all_rows.k.eq(20)]
        method=selected.method.iloc[0]
        paths=selected.folder.map(lambda p:p.relative_to(ROOT).as_posix())
        # The saved runtime table uses the original shard/run layout.
        saved = runtime.loc[["/".join(Path(p).parts[-2:]) for p in paths]]
        metrics = [next(v for v in json.loads((p/"metrics.json").read_text()) if v["domain"]=="target_test")
                   for p in selected.folder]
        key = {"conv":"conv","mlp":"mlp","norm":"conv","no_replay":"conv","dino":"dino","patchcore":"cnn"}[method]
        score_field = "score_image_seconds" if key in ("conv","mlp") else "knn_image_seconds"
        adapt = "decoder" if variant in ("conv","mlp") else variant
        update = [t[key].get(adapt+"_two_steps_seconds") for t in timings]
        one_update = pd.Series([v/2 for v in update if v is not None],dtype=float).median() if any(v is not None for v in update) else math.nan
        scoring = pd.Series([t[key][score_field] for t in timings],dtype=float).median() if timings else math.nan
        bank_fit = [t[key]["memory_fit_8192_candidates_seconds"] for t in timings if key in ("cnn","dino")]
        records.append(dict(variant=variant,method=name,k=20,runs=len(selected),
                            detector_parameters=metrics[0].get("parameters",0),
                            backbone_parameters="not available: frozen backbone parameter count unlogged",
                            trainable_parameters=metrics[0]["trainable_parameters"],
                            fit_seconds_median=saved.fit_seconds.median(),
                            elapsed_seconds_median=saved.elapsed_seconds.median(),
                            smoke_bank_fit_seconds=pd.Series(bank_fit,dtype=float).median(),
                            smoke_seconds_per_update=one_update,
                            smoke_seconds_per_cached_image=scoring,
                            peak_allocated_MiB=saved.peak_allocated_bytes.max()/2**20,
                            peak_reserved_MiB=saved.peak_reserved_bytes.max()/2**20,
                            memory_bank_patches=metrics[0].get("bank_size",math.nan),
                            timing_scope="K20 fit interval includes loading/fitting/saving. Per-image score and per-update times are saved source-only preflight medians, not end-to-end test timings.",
                            detector_parameters_note="Memory methods have no fitted neural head; frozen backbone parameter count unlogged." if "parameters" not in metrics[0] else "Trainable reconstruction network only; frozen backbone excluded.",
                            source_image_entries=source_n))
    out.csv(out.comparison/"tables","report-efficiency-k20",pd.DataFrame(records))


def spot_check(out, rows, condition, labels):
    candidates = ["image_auroc","image_auprc","fpr_at_95_tpr",*RATES]
    cells = [(i,m) for i in range(len(condition)) for m in candidates
             if pd.notna(condition.iloc[i][m+"_mean"])]
    checked = []
    for index, metric in random.Random(20261112).sample(cells,10):
        c = condition.iloc[index]
        group = rows[(rows.variant==c.variant)&(rows.condition==c.condition)&(rows.k==c.k)]
        values = []
        for r in group.to_dict("records"):
            test = read_scores(r,labels)["target_test"]
            values.append((binary_counts(test.label,test.score,r["threshold"]) | ranking(test.label,test.score)[0])[metric])
        mean = pd.Series(values).mean()
        assert close(mean,c[metric+"_mean"]),(c.variant,c.condition,c.k,metric,mean)
        checked.append(dict(variant=c.variant,condition=c.condition,k=c.k,metric=metric,
                            table_value=c[metric+"_mean"],recomputed=mean,match=True))
    out.csv(out.comparison/"tables","report-consistency",pd.DataFrame(checked))
    return len(checked)


def summary(out, macro, tests, checked, statistics):
    inference,q2,q3 = statistics
    lo,hi = inference["ci95_difference"]
    cutoff = q3[q3.variant.eq("patchcore")].set_index("policy")
    compact = macro.loc[macro.variant.isin(MAIN),["variant","k","seeds","normal_n_mean","anomaly_n_mean",
                         "image_auroc_mean","image_auroc_std","balanced_accuracy_mean","balanced_accuracy_std"]]
    out.csv(out.comparison/"tables","report-headline",compact)
    counts=tests.groupby("condition").label.agg(["size","sum"])
    lines = [
        "# MetalParts reporting summary", "",
        "All results use the 448 px saved predictions and the frozen MetalParts evaluation sets.",
        "Headline metrics are **balanced accuracy and image AUROC**. Accuracy can hide class imbalance.",
        "Positive class is anomalous. TP = detected anomaly, FN = missed anomaly, FP = false alarm, TN = correct normal.",
        "At K=0/5/20, score > the saved source-normal 95th-quantile threshold predicts anomaly.",
        "These decisions are labelled **source-fixed threshold under score shift**.",
        "Q3 uses the two saved thresholds on the same 10-adapted detector, with ten separate target calibration normals.",
        "Counts are calculated per condition and seed. The sum adds conditions within each seed before averaging.",
        "Rates are per-run ratios: TPR = TP/(TP+FN), TNR = TN/(TN+FP), FPR = 1−TNR, FNR = 1−TPR.",
        "Precision = TP/(TP+FP), accuracy = (TP+TN)/n, balanced accuracy = (TPR+TNR)/2, F1 = 2TP/(2TP+FP+FN).",
        "Precision is unavailable when no positive is predicted. Tables include the number of defined seed values.",
        "Macro means weight six conditions equally. Per-condition and macro SDs are descriptive sample SDs across seeds.",
        "AUROC and AUPRC (average precision) are computed from saved scores. FPR@95%TPR uses the smallest achieved FPR at TPR ≥95%.",
        "ROC figures average seed and condition curves on a 201-point FPR grid using interpolation for display.",
        "Pixel AUROC and AUPRO (FPR ≤0.30) use the saved 448×448 evaluations: bilinear score maps, nearest-neighbor masks, exact tied-score region weighting.",
        "Score overlays compare within-seed, within-condition percentile ranks. Per-model plots use recorded raw scores.",
        "Examples use the first eligible saved-map image ID per outcome and condition, breaking ties by seed.",
        "Shared example figures use Conv AE outcomes and show the same image and seed for every method.", "",
        "| Target test denominators | Normal | Anomalous |", "|---|---:|---:|",
    ]
    for condition,r in counts.iterrows():
        lines.append(f"| {condition} | {int(r['size']-r['sum'])} | {int(r['sum'])} |")
    lines += ["","| Method | K | Image AUROC mean ± SD | Balanced accuracy mean ± SD |",
              "|---|---:|---:|---:|"]
    for method in MAIN:
        for r in compact[compact.variant.eq(method)].sort_values("k").itertuples():
            lines.append(f"| {METHODS[method][0]} | {r.k} | {r.image_auroc_mean:.3f} ± {r.image_auroc_std:.3f} | {r.balanced_accuracy_mean:.3f} ± {r.balanced_accuracy_std:.3f} |")
    lines += ["","DINOv2 leads on K=20 AUROC. Conv AE's AUROC changes little from K=5 to K=20.",
              f"At K=20 the matched MLP reaches {q2.at['mlp','mean']:.3f} AUROC versus {q2.at['conv','mean']:.3f} for Conv AE.",
              f"PatchCore calibration changes false alarms from {cutoff.at[FIXED,'fpr_mean']:.1%} to {cutoff.at[RECAL,'fpr_mean']:.1%} and recall from {cutoff.at[FIXED,'recall_mean']:.1%} to {cutoff.at[RECAL,'recall_mean']:.1%}.",
              f"All nine headline AUROCs matched. All {checked} table cells matched a recomputation from saved scores.",
              "The memory variants use deterministic K=0 aliases, so their K=0 seed SD is zero.",
              f"The saved Conv AE minus DINOv2 K=20 difference is {inference['difference']:.3f} (95% CI {lo:.3f} to {hi:.3f}). Predeclared permutation p={inference['permutation_p']:.6f}.",
              "", "![Macro AUROC](figures/report-macro-auroc-bars.png)",
              "![K=20 confusion](figures/report-confusion-k20.png)",
              "![Ablations](figures/report-q2-ablations.png)",
              "![Calibration](figures/report-q3-calibration.png)",
              "![Paired difference](figures/report-k20-paired-effects.png)",
              "", "**Limitations.** Six development conditions and small, imbalanced test sets.",
              "Capture identity is unverified. Image-level independence is assumed and the intervals may be optimistic.",
              "Source thresholds use 11–14 normal images, support draws overlap across seeds, and score shift affects operating rates.",
              "Ten calibration normals give a coarse estimate. Saved maps have no false-alarm examples. Panels mark missing examples.",
              "The report uses the saved interval and permutation result for the predeclared pair.",
              "Scoring latency comes from a small saved preflight on cached source features."]
    assert len(lines)<=60,len(lines)
    path=out.comparison/"summary.md";path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    out.paths.append(path)


def generate(out_dir,manifest_dir):
    comparison=Path(out_dir).resolve()
    labels,paths=labels_and_inputs(Path(manifest_dir))
    records=evidence()
    headline_gate(records,labels)
    all_rows, raw_roc, scores=measure(records,labels)
    rows=all_rows[all_rows.queue.ne("Q3")]
    metrics=RATES+RANKING+RETENTION+DENOM
    condition=aggregate(rows,KEYS+["condition"],metrics)
    seed_macro=rows.groupby(KEYS+["seed"],as_index=False)[metrics].mean()
    # Sum the fixed condition denominators; performance metrics use equal-weight means.
    totals=rows.groupby(KEYS+["seed"])[DENOM].sum()
    seed_macro=seed_macro.set_index(KEYS+["seed"])
    seed_macro[DENOM]=totals
    macro=aggregate(seed_macro.reset_index(),KEYS,metrics)
    counts=mean_counts(rows,KEYS+["condition"])
    summed=mean_counts(rows.groupby(KEYS+["seed"],as_index=False)[COUNTS+DENOM].sum(),KEYS).assign(condition="sum")
    counts=pd.concat([counts,summed],ignore_index=True)
    roc=raw_roc.groupby(["variant","k","condition","fpr"]).tpr.mean().groupby(["variant","k","fpr"]).mean().reset_index()
    condition["threshold_label"]=FIXED
    condition["accuracy_note"]="class-imbalanced; prefer balanced accuracy and AUROC"
    condition["pixel_status"] = condition.apply(lambda r:"Saved full-mask metrics" if r.aupro_count==r.seeds and r.pixel_auroc_count==r.seeds else "not available for some seeds: check per-seed mask status",axis=1)
    condition["n_images"]=condition.normal_n_mean+condition.anomaly_n_mean
    macro["accuracy_note"]="class-imbalanced; prefer balanced accuracy and AUROC"
    targets=labels.xs("target_test",level="role").reset_index()
    out=Outputs(comparison);styling()
    checked=spot_check(out,rows,condition,labels)
    out.csv(comparison/"tables","report-full-metrics",condition)
    out.csv(comparison/"tables","report-full-confusion",counts)
    out.csv(comparison/"tables","report-macro",macro)
    out.csv(comparison/"tables","report-roc-grid",roc)
    per_model(out,rows,condition,macro,counts,roc,scores,paths)
    statistics=figure_comparison(out,all_rows,condition,macro,counts,roc,scores,paths)
    summary(out,macro,targets,checked,statistics)
    print("PASS: 10 random aggregate cells match saved predictions",flush=True)
    print("CREATED",json.dumps(Counter(p.parent.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p.parent) for p in out.paths)),flush=True)
    print("UNAVAILABLE QUALITATIVE SLOTS",json.dumps(out.missing),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", required=True, help="Generate tables and figures from saved results.")
    parser.parse_args()
    generate(ROOT/"comparison",ROOT/"data/manifests/MetalParts")
