"""Declare a controlled matrix before any result is read; run with --execute."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from .train import run


def matrix():
    base={"p":97,"fraction":.6,"val_fraction":.2,"width":32,"lr":.003,
          "weight_decay":1.,"steps":3000,"batch_size":256,"eval_every":100,
          "threads":1,"device":"cpu"}
    jobs=[]
    for model in ["mlp","transformer"]:
        conditions=[("main",{})]
        if model=="transformer":conditions += [("fraction",{"fraction":.3}),("decay",{"weight_decay":0.})]
        for condition,override in conditions:
            for split in ["ordered","grouped"]:
                for seed in [0,1,2]:
                    config={**base,**override,"model":model,"split":split,"seed":seed,"condition":condition}
                    name=f"{model}_{condition}_{split}_s{seed}"
                    jobs.append((name,config))
    return jobs


def worker(job):
    name,config,out=job
    summary=run(config,out)
    return name,summary["final_train_accuracy"],summary["test"]["accuracy"],summary["seconds"]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--root",default=".");args=ap.parse_args();root=Path(args.root)
    configs=root/"configs";configs.mkdir(parents=True,exist_ok=True)
    jobs=matrix()
    declaration=[{"name":n,"config":c} for n,c in jobs]
    manifest=configs/"executed_matrix.json"
    if manifest.exists():
        if json.loads(manifest.read_text())!=declaration:raise RuntimeError("Declared matrix changed")
    else:manifest.write_text(json.dumps(declaration,indent=2))
    for name,c in jobs:(configs/f"{name}.json").write_text(json.dumps(c,indent=2))
    print(f"Declared {len(jobs)} runs; final test evaluation at fixed step 3000",flush=True)
    if not args.execute:return
    pending=[(name,c,str(root/"results"/name)) for name,c in jobs if not (root/"results"/name/"summary.json").exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,j) for j in pending]
        for f in concurrent.futures.as_completed(futures):
            name,tr,te,secs=f.result()
            print(f"{name}: train={tr:.3%}, test={te:.3%}, seconds={secs:.1f}",flush=True)


if __name__=="__main__":main()
