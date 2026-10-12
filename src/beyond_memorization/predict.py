import argparse
import torch
from .models import build_model


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("a",type=int);ap.add_argument("b",type=int)
    args=ap.parse_args();saved=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    c=saved["config"];p=c["p"]
    if not (0<=args.a<p and 0<=args.b<p):ap.error(f"Operands must lie in [0,{p-1}]")
    model=build_model(c["model"],p,c["width"]);model.load_state_dict(saved["state_dict"]);model.eval()
    with torch.no_grad():answer=int(model(torch.tensor([[args.a,args.b]])).argmax(-1))
    print(f"Model prediction: {answer}; arithmetic target: {(args.a+args.b)%p}; p={p}")


if __name__=="__main__":main()
