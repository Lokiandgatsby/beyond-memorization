"""Fixed-horizon training. Test data are evaluated only after training ends."""
import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch.nn import functional as F
from .data import make_split,audit,swapped_mask,lookup_baseline
from .models import build_model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def metrics(model,x,y,batch=2048):
    model.eval(); total_loss=0.; preds=[]
    for start in range(0,len(x),batch):
        logits=model(x[start:start+batch]); target=y[start:start+batch]
        total_loss+=F.cross_entropy(logits,target,reduction="sum").item()
        preds.append(logits.argmax(-1).cpu().numpy())
    prediction=np.concatenate(preds)
    return {"loss":total_loss/len(x),"accuracy":float((prediction==y.cpu().numpy()).mean())},prediction


def run(config,out):
    out=Path(out)
    if (out/"summary.json").exists(): raise FileExistsError(f"Refuse to overwrite completed run {out}")
    out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    torch.set_num_threads(config.get("threads",1)); seed_all(config["seed"])
    device=torch.device(config.get("device","cpu"))
    env={"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
         "platform":platform.platform(),"device":str(device),"threads":torch.get_num_threads(),
         "cpu":platform.processor(),"started_utc":datetime.now(timezone.utc).isoformat()}
    if device.type=="cuda":env["gpu"]=torch.cuda.get_device_name(device)
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    p=config["p"]
    parts=make_split(p,config["fraction"],config["val_fraction"],config["seed"],config["split"])
    np.savez_compressed(out/"split.npz",**parts)
    (out/"split_audit.json").write_text(json.dumps(audit(parts,p),indent=2))
    # Only training and validation tensors are exposed during optimization.
    x={k:torch.tensor(parts[k],device=device) for k in ["train","val"]}
    y={k:x[k].sum(-1)%p for k in x}
    model=build_model(config["model"],p,config["width"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],betas=(.9,.98),
                               weight_decay=config["weight_decay"])
    generator=torch.Generator(device=device).manual_seed(config["seed"]+10000)
    log=[]; start=time.perf_counter(); best_val=float("inf"); best_step=0
    with (out/"history.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["step","train_loss","train_accuracy","val_loss","val_accuracy","weight_norm"])
        writer.writeheader()
        for step in range(config["steps"]+1):
            if step%config["eval_every"]==0 or step==config["steps"]:
                tr,_=metrics(model,x["train"],y["train"]); va,_=metrics(model,x["val"],y["val"])
                row={"step":step,"train_loss":tr["loss"],"train_accuracy":tr["accuracy"],
                     "val_loss":va["loss"],"val_accuracy":va["accuracy"],
                     "weight_norm":sum(t.detach().square().sum().item() for t in model.parameters())**.5}
                writer.writerow(row); f.flush(); log.append(row)
                if va["loss"]<best_val:best_val=va["loss"];best_step=step
            if step==config["steps"]:break
            model.train(); optimizer.zero_grad(set_to_none=True)
            bs=config["batch_size"]
            if bs==0:inputs=x["train"];targets=y["train"]
            else:
                idx=torch.randint(len(x["train"]),(bs,),generator=generator,device=device)
                inputs=x["train"][idx]; targets=y["train"][idx]
            loss=F.cross_entropy(model(inputs),targets)
            loss.backward(); optimizer.step()
    # Final checkpoint is fixed in advance; no test-guided selection.
    torch.save({"state_dict":model.state_dict(),"config":config},out/"final.pt")
    xt=torch.tensor(parts["test"],device=device);yt=xt.sum(-1)%p
    test,pred=metrics(model,xt,yt)
    swapped=swapped_mask(parts["test"],parts["train"])
    def subgroup(mask):
        n=int(mask.sum())
        return {"n":n,"accuracy":float((pred[mask]==yt.cpu().numpy()[mask]).mean()) if n else None}
    np.savez_compressed(out/"test_predictions.npz",pairs=parts["test"],target=yt.cpu().numpy(),prediction=pred,swapped_seen=swapped)
    @torch.no_grad()
    def consistency():
        pa=[];pb=[]
        allx=torch.tensor([(a,b) for a in range(p) for b in range(p)],device=device)
        for chunk in allx.split(2048):
            pa.append(model(chunk).argmax(-1));pb.append(model(chunk.flip(-1)).argmax(-1))
        return float((torch.cat(pa)==torch.cat(pb)).float().mean())
    train_time=next((r["step"] for r in log if r["train_accuracy"]>=.99),None)
    val_time=next((r["step"] for r in log if r["val_accuracy"]>=.9),None)
    summary={"config":config,"parameters":sum(t.numel() for t in model.parameters()),
             "seconds":time.perf_counter()-start,"final_train_accuracy":log[-1]["train_accuracy"],
             "final_val_accuracy":log[-1]["val_accuracy"],"test":test,
             "test_swapped_seen":subgroup(swapped),"test_swapped_unseen":subgroup(~swapped),
             "prediction_swap_consistency":consistency(),"baseline":lookup_baseline(parts,p),
             "train_99_first":train_time,"val_90_first":val_time,
             "delayed_generalization_proxy":bool(train_time is not None and val_time is not None and val_time-train_time>=500),
             "best_val_loss_step_diagnostic_only":best_step}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--out",required=True)
    args=ap.parse_args();s=run(json.loads(Path(args.config).read_text()),args.out)
    print(json.dumps(s,indent=2))


if __name__=="__main__":main()
