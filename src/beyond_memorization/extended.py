"""Explicitly post-hoc longer-horizon diagnostic; all six base settings included."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from .suite import matrix,worker


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--root",default=".")
    ap.add_argument("--declare-only",action="store_true")
    args=ap.parse_args()
    root=Path(args.root)
    (root/"configs").mkdir(parents=True,exist_ok=True)
    declaration=[]
    for name,c in matrix():
        if c["model"]=="transformer" and c["condition"]=="main":
            c={**c,"steps":12000,"condition":"horizon","eval_every":100}
            declaration.append({"name":name.replace("main","horizon"),"config":c})
    manifest=root/"configs/exploratory_matrix.json"
    if manifest.exists() and json.loads(manifest.read_text())!=declaration:raise RuntimeError("Configuration drift")
    manifest.write_text(json.dumps(declaration,indent=2))
    pending=[]
    for entry in declaration:
        name=entry["name"];c=entry["config"]
        (root/"configs"/f"{name}.json").write_text(json.dumps(c,indent=2))
        folder=root/"results"/name
        if not (folder/"summary.json").exists():pending.append((name,c,str(folder)))
    if args.declare_only:
        print(f"Declared {len(declaration)} exploratory configurations under {root}")
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,pending):print(result,flush=True)


if __name__=="__main__":main()
