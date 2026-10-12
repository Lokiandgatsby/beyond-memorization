"""Check saved scientific artifacts against their raw arrays and checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .analyze import load_runs
from .data import validate_split,audit,swapped_mask
from .models import build_model


def verify(root):
    root=Path(root);torch.set_num_threads(1);runs=load_runs(root)
    for r in runs:
        folder=root/"results"/r["name"];s=r["summary"];c=s["config"]
        parts=dict(np.load(folder/"split.npz"));validate_split(parts,c["p"],c["split"]=="grouped")
        assert audit(parts,c["p"])==r["audit"]
        pred=np.load(folder/"test_predictions.npz")
        np.testing.assert_array_equal(pred["pairs"],parts["test"])
        np.testing.assert_array_equal(pred["target"],parts["test"].sum(-1)%c["p"])
        np.testing.assert_array_equal(pred["swapped_seen"],swapped_mask(parts["test"],parts["train"]))
        assert float((pred["prediction"]==pred["target"]).mean())==s["test"]["accuracy"]
        saved=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
        assert saved["config"]==c
        model=build_model(c["model"],c["p"],c["width"]);model.load_state_dict(saved["state_dict"]);model.eval()
        with torch.no_grad():
            chunks=torch.tensor(parts["test"]).split(2048)
            actual=np.concatenate([model(chunk).argmax(-1).numpy() for chunk in chunks])
        np.testing.assert_array_equal(actual,pred["prediction"])
    extensions=[r for r in runs if r["summary"]["config"]["condition"]=="horizon"]
    for r in extensions:
        original=next(v for v in runs if v["name"]==r["name"].replace("horizon","main"))
        orig_history=original["history"]
        np.testing.assert_array_equal(r["history"][:len(orig_history)],orig_history)
    output={"runs_verified":len(runs),"checkpoint_predictions_match":True,
            "split_integrity":True,"extended_prefix_matches_base":len(extensions),
            "test_accuracy_recomputed_from_saved_arrays":True}
    (root/"results/artifact_verification.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");args=ap.parse_args();verify(args.root)


if __name__=="__main__":main()
