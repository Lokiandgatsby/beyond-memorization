"""Two deliberately small classifiers; no arithmetic features are supplied."""
import math
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,p,width=32):
        super().__init__()
        self.embedding=nn.Embedding(p,width)
        self.network=nn.Sequential(nn.Linear(2*width,4*width),nn.GELU(),
                                   nn.Linear(4*width,4*width),nn.GELU(),nn.Linear(4*width,p))
    def forward(self,pairs):
        return self.network(self.embedding(pairs).flatten(1))


class TinyTransformer(nn.Module):
    """One causal pre-norm attention block over [a,b,=], with an MLP."""
    def __init__(self,p,width=32,heads=4):
        super().__init__()
        if width%heads: raise ValueError("width must divide into heads")
        self.p=p; self.heads=heads; self.width=width
        self.embedding=nn.Embedding(p+1,width)
        self.position=nn.Parameter(torch.randn(3,width)*.02)
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width); self.proj=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.final_norm=nn.LayerNorm(width); self.unembed=nn.Linear(width,p)
        self.register_buffer("mask",torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1))
    def forward(self,pairs):
        tokens=torch.cat([pairs,torch.full((len(pairs),1),self.p,device=pairs.device,dtype=torch.long)],dim=1)
        x=self.embedding(tokens)+self.position
        qkv=self.qkv(self.norm1(x)).reshape(len(x),3,3,self.heads,self.width//self.heads)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-1,-2))/math.sqrt(self.width//self.heads)
        attention=scores.masked_fill(self.mask,float("-inf")).softmax(dim=-1)
        out=(attention@v).transpose(1,2).reshape(len(x),3,self.width)
        x=x+self.proj(out); x=x+self.ff(self.norm2(x))
        return self.unembed(self.final_norm(x[:,-1]))


def build_model(name,p,width=32):
    if name=="mlp": return MLP(p,width)
    if name=="transformer": return TinyTransformer(p,width)
    raise ValueError(name)
