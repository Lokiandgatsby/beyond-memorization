"""Derive every figure and table from saved results, never synthetic curves."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"ordered":"#2563eb","grouped":"#d97706"}


def load_runs(root):
    declaration=json.loads((root/"configs/executed_matrix.json").read_text())
    extension=root/"configs/exploratory_matrix.json"
    if extension.exists():declaration+=json.loads(extension.read_text())
    runs=[]
    for entry in declaration:
        folder=root/"results"/entry["name"]
        summary=json.loads((folder/"summary.json").read_text())
        if summary["config"]!=entry["config"]:raise ValueError("Configuration drift")
        history=np.genfromtxt(folder/"history.csv",delimiter=",",names=True)
        if int(history[-1]["step"])!=entry["config"]["steps"]:raise ValueError("Incomplete run")
        runs.append({"name":entry["name"],"summary":summary,"history":history,
                     "audit":json.loads((folder/"split_audit.json").read_text())})
    return runs


def analyze(root):
    root=Path(root);fig=root/"figures";fig.mkdir(exist_ok=True)
    runs=load_runs(root);groups=defaultdict(list)
    for r in runs:
        c=r["summary"]["config"];groups[c["model"],c["condition"],c["split"]].append(r)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":160,"savefig.bbox":"tight"})
    aggregated=[]
    for key,rs in groups.items():
        values=np.array([r["summary"]["test"]["accuracy"] for r in rs])
        sds=lambda values:float(np.std(values,ddof=1))
        aggregated.append({"model":key[0],"condition":key[1],"split":key[2],"n":len(rs),
                           "test_mean":float(values.mean()),"test_sd":sds(values),
                           "train_mean":float(np.mean([r["summary"]["final_train_accuracy"] for r in rs])),
                           "val_mean":float(np.mean([r["summary"]["final_val_accuracy"] for r in rs])),
                           "lookup_mean":float(np.mean([r["summary"]["baseline"]["swap_lookup_accuracy"] for r in rs])),
                           "overlap_mean":float(np.mean([r["audit"]["test"]["non_diagonal_swapped_rate"] for r in rs]))})
    (root/"results/aggregated.json").write_text(json.dumps(aggregated,indent=2))
    with (root/"results/aggregated.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=aggregated[0]);writer.writeheader();writer.writerows(aggregated)
    rows=[]
    for r in runs:
        s=r["summary"];c=s["config"]
        rows.append({"run":r["name"],**{k:c[k] for k in ["model","condition","split","seed","fraction","weight_decay","steps"]},
                     "train_accuracy":s["final_train_accuracy"],"val_accuracy":s["final_val_accuracy"],
                     "test_accuracy":s["test"]["accuracy"],"test_loss":s["test"]["loss"],
                     "parameters":s["parameters"],"seconds":s["seconds"],
                     "train_99_first":s["train_99_first"],"val_90_first":s["val_90_first"],
                     "swapped_seen_n":s["test_swapped_seen"]["n"],"swapped_seen_acc":s["test_swapped_seen"]["accuracy"],
                     "swapped_unseen_acc":s["test_swapped_unseen"]["accuracy"],
                     "swap_consistency":s["prediction_swap_consistency"]})
    with (root/"results/all_runs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Figure 1: per-seed train/validation curves; shaded sample SD.
    figure,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for i,model in enumerate(["mlp","transformer"]):
        for j,split in enumerate(["ordered","grouped"]):
            ax=axes[i,j];rs=groups[model,"main",split]
            for field,label,style in [("train_accuracy","Train","-"),("val_accuracy","Validation","--")]:
                arr=np.stack([r["history"][field] for r in rs]);steps=rs[0]["history"]["step"]
                avg=arr.mean(0);sd=arr.std(0,ddof=1)
                color=COLORS[split] if field=="val_accuracy" else "#334155"
                ax.plot(steps,avg,label=label,linestyle=style,color=color)
                ax.fill_between(steps,np.clip(avg-sd,0,1),np.clip(avg+sd,0,1),alpha=.15,color=color)
            ax.set_title(f"{model.upper()} / {split}");ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.legend(loc="upper left")
            if i==1:ax.set_xlabel("Optimizer updates")
            if j==0:ax.set_ylabel("Accuracy")
    figure.suptitle("Learning dynamics at p=97, train fraction 0.6, weight decay 1",fontsize=13)
    figure.tight_layout();figure.savefig(fig/"learning_curves.png");plt.close(figure)
    # Figure 2: final evaluation for all ablations. Points show actual seeds.
    cases=[("mlp","main","MLP / base"),("transformer","main","Transformer / base"),
           ("transformer","fraction","Transformer / 30% train"),("transformer","decay","Transformer / no decay")]
    figure,ax=plt.subplots(figsize=(10,4.5))
    for j,split in enumerate(["ordered","grouped"]):
        for i,(model,condition,label) in enumerate(cases):
            rs=groups[model,condition,split];vals=np.array([r["summary"]["test"]["accuracy"] for r in rs])
            x=i+(j-.5)*.32
            ax.bar(x,vals.mean(),width=.28,color=COLORS[split],alpha=.8,label=split if i==0 else None)
            ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt="none",color="#111827",capsize=4)
            ax.scatter(x+np.linspace(-.045,.045,3),vals,color="#111827",s=16,zorder=5)
    ax.set_xticks(range(4),[c[2] for c in cases]);ax.set_ylim(0,1.05);ax.set_ylabel("Final test accuracy")
    ax.set_title("Fixed 3,000-update horizon; bars: mean, error bars: sample SD, dots: seeds")
    ax.legend();ax.grid(axis="y",alpha=.2);figure.tight_layout();figure.savefig(fig/"ablation_comparison.png");plt.close(figure)
    # Figure 3: strong control showing the evaluation protocol admits swap retrieval.
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for i,split in enumerate(["ordered","grouped"]):
        rs=groups["transformer","main",split]
        arrays=[[r["audit"]["test"]["non_diagonal_swapped_rate"] for r in rs],
                [r["summary"]["baseline"]["swap_lookup_accuracy"] for r in rs],
                [r["summary"]["test"]["accuracy"] for r in rs]]
        ax=axes[i];means=[np.mean(a) for a in arrays];sds=[np.std(a,ddof=1) for a in arrays]
        ax.bar(range(3),means,yerr=sds,capsize=4,color=["#64748b","#059669",COLORS[split]])
        ax.set_xticks(range(3),["Swap overlap*","Swap lookup","Transformer"]);ax.set_title(split);ax.set_ylim(0,1.05)
        if i==0:ax.set_ylabel("Rate / accuracy")
        ax.grid(axis="y",alpha=.2)
    figure.suptitle("*Overlap denominator excludes diagonal pairs; accuracies use all test pairs")
    figure.tight_layout();figure.savefig(fig/"swap_controls.png");plt.close(figure)
    # Figure 4: test-only error maps, predetermined seed 0 (no model cherry-picking).
    figure,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,split in zip(axes,["ordered","grouped"]):
        stored=np.load(root/f"results/transformer_main_{split}_s0/test_predictions.npz")
        grid=np.full((97,97),np.nan)
        pairs=stored["pairs"];grid[pairs[:,0],pairs[:,1]]=stored["prediction"]==stored["target"]
        cmap=plt.get_cmap("RdYlGn").copy();cmap.set_bad("#e2e8f0")
        im=ax.imshow(grid,cmap=cmap,vmin=0,vmax=1,origin="lower",interpolation="nearest")
        ax.set_xlabel("b");ax.set_ylabel("a");ax.set_title(f"Transformer / {split} / seed 0")
    figure.suptitle("Final test predictions: green correct, red wrong, gray not in test")
    figure.tight_layout();figure.savefig(fig/"test_error_maps.png");plt.close(figure)
    if ("transformer","horizon","ordered") in groups:
        figure,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
        for ax,split in zip(axes,["ordered","grouped"]):
            rs=groups["transformer","horizon",split]
            for field,label,style in [("train_accuracy","Train","-"),("val_accuracy","Validation","--")]:
                arr=np.stack([r["history"][field] for r in rs]);steps=rs[0]["history"]["step"]
                color=COLORS[split] if field=="val_accuracy" else "#334155"
                ax.plot(steps,arr.mean(0),color=color,linestyle=style,label=label)
                ax.fill_between(steps,np.clip(arr.mean(0)-arr.std(0,ddof=1),0,1),np.clip(arr.mean(0)+arr.std(0,ddof=1),0,1),color=color,alpha=.15)
            ax.axvline(3000,color="#64748b",linestyle=":",label="Main horizon")
            ax.set_title(split);ax.set_xlabel("Optimizer updates");ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel("Accuracy");figure.suptitle("Exploratory 12,000-update diagnostic; not a fresh confirmatory study")
        figure.tight_layout();figure.savefig(fig/"extended_dynamics.png");plt.close(figure)
    # Paired differences, with no unsupported confidence intervals.
    paired=[]
    for model,cond,_ in cases:
        ordered=sorted(groups[model,cond,"ordered"],key=lambda r:r["summary"]["config"]["seed"])
        grouped=sorted(groups[model,cond,"grouped"],key=lambda r:r["summary"]["config"]["seed"])
        diffs=[g["summary"]["test"]["accuracy"]-o["summary"]["test"]["accuracy"] for o,g in zip(ordered,grouped)]
        paired.append({"model":model,"condition":cond,"grouped_minus_ordered":diffs,"mean":float(np.mean(diffs)),"sd":float(np.std(diffs,ddof=1))})
    stats={"runs":len(runs),"summed_run_seconds":sum(r["summary"]["seconds"] for r in runs),
           "proxy_count":sum(r["summary"]["delayed_generalization_proxy"] for r in runs),"paired_differences":paired,
           "train_99_count":sum(r["summary"]["train_99_first"] is not None for r in runs),
           "val_90_count":sum(r["summary"]["val_90_first"] is not None for r in runs)}
    (root/"results/study_statistics.json").write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
    return runs,aggregated,stats


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");args=ap.parse_args();analyze(args.root)


if __name__=="__main__":main()
