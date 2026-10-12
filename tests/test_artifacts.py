import json
import numpy as np
import torch
from beyond_memorization.train import run
from beyond_memorization.models import build_model


def test_saved_checkpoint_reproduces_final_predictions(tmp_path):
    cfg={"p":5,"fraction":.6,"val_fraction":.2,"width":8,"lr":.003,"weight_decay":1.,
         "steps":4,"batch_size":8,"eval_every":2,"threads":1,"device":"cpu", "seed":0,
         "model":"transformer","split":"grouped","condition":"test"}
    summary=run(cfg,tmp_path)
    saved=torch.load(tmp_path/"final.pt",weights_only=True)
    model=build_model("transformer",5,8);model.load_state_dict(saved["state_dict"]);model.eval()
    pred=np.load(tmp_path/"test_predictions.npz")
    with torch.no_grad():actual=model(torch.tensor(pred["pairs"])).argmax(-1).numpy()
    np.testing.assert_array_equal(actual,pred["prediction"])
    assert summary["test"]["accuracy"]==float((actual==pred["target"]).mean())
    assert json.loads((tmp_path/"config.json").read_text())==cfg
    assert (tmp_path/"history.csv").read_text().count("\n")==4
