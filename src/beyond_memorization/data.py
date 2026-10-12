"""Exact-count partitions. Grouped partitions keep swap orbits intact."""
import hashlib
import numpy as np


def make_split(p=97, fraction=.6, val_fraction=.2, seed=0, strategy="ordered"):
    if p < 3 or not (0 < fraction < 1 and 0 < val_fraction < 1-fraction):
        raise ValueError("Require p>=3 and nonempty train/validation/test proportions")
    if strategy not in {"ordered", "grouped"}:
        raise ValueError("Unknown split strategy")
    n = p*p
    counts = [int(n*fraction), int(n*val_fraction)]
    counts.append(n-sum(counts))
    if min(counts) < 1:
        raise ValueError("Empty partition")
    rng = np.random.default_rng(seed)
    all_pairs = np.array([(a,b) for a in range(p) for b in range(p)], dtype=np.int64)
    if strategy == "ordered":
        indices = rng.permutation(n)
        chunks = np.split(indices, np.cumsum(counts)[:-1])
        parts = [all_pairs[i] for i in chunks]
    else:
        # Match exact ordered-pair counts using singleton diagonal orbits.
        # Choose diagonal allocations closest to proportional allocation.
        candidates = []
        for d0 in range(min(p,counts[0])+1):
            if (counts[0]-d0)%2: continue
            for d1 in range(min(p-d0,counts[1])+1):
                d2 = p-d0-d1
                ds = [d0,d1,d2]
                if all(0<=d<=c and (c-d)%2==0 for d,c in zip(ds,counts)):
                    score = sum((d-c/p)**2 for d,c in zip(ds,counts))
                    candidates.append((score,ds))
        if not candidates: raise ValueError("No exact grouped allocation exists")
        ds = min(candidates, key=lambda t:t[0])[1]
        diag = rng.permutation(p)
        off = np.array([(a,b) for a in range(p) for b in range(a+1,p)],dtype=np.int64)
        off = off[rng.permutation(len(off))]
        parts=[]; di=0; oi=0
        for c,d in zip(counts,ds):
            k=(c-d)//2
            orbits=off[oi:oi+k]; singles=diag[di:di+d]
            pairs=np.concatenate([orbits,orbits[:,::-1],np.column_stack([singles,singles])])
            parts.append(pairs[rng.permutation(len(pairs))])
            di+=d; oi+=k
    result = dict(zip(["train","val","test"],parts))
    validate_split(result,p,grouped=(strategy=="grouped"))
    return result


def validate_split(parts,p,grouped=False):
    sets = {k:set(map(tuple,v.tolist())) for k,v in parts.items()}
    assert all(len(sets[k])==len(parts[k]) for k in sets), "Duplicate pair"
    assert sum(map(len,sets.values()))==p*p
    assert set.union(*sets.values())=={(a,b) for a in range(p) for b in range(p)}
    keys=list(sets)
    for i,k in enumerate(keys):
        for other in keys[i+1:]:
            assert sets[k].isdisjoint(sets[other]), "Exact-pair contamination"
            if grouped:
                assert {tuple(sorted(t)) for t in sets[k]}.isdisjoint(
                    {tuple(sorted(t)) for t in sets[other]}), "Swap orbit crosses partitions"


def swapped_mask(pairs,train):
    seen=set(map(tuple,train.tolist()))
    return np.array([(int(b),int(a)) in seen for a,b in pairs],dtype=bool)


def audit(parts,p):
    output={"counts":{k:len(v) for k,v in parts.items()},
            "sha256":{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in parts.items()}}
    for name in ["val","test"]:
        pairs=parts[name]; non=pairs[:,0]!=pairs[:,1]
        overlap=swapped_mask(pairs,parts["train"])
        output[name]={"non_diagonal_n":int(non.sum()),"swapped_seen_n":int(overlap.sum()),
                      "non_diagonal_swapped_rate":float(overlap[non].mean()) if non.any() else None,
                      "class_counts":np.bincount(pairs.sum(axis=1)%p,minlength=p).tolist()}
    output["diagonal_counts"]={k:int((v[:,0]==v[:,1]).sum()) for k,v in parts.items()}
    return output


def lookup_baseline(parts,p):
    train=parts["train"]
    lookup={tuple(t):int(t.sum()%p) for t in train}
    fallback=int(np.bincount(train.sum(axis=1)%p,minlength=p).argmax())
    def evaluate(pairs,swaps):
        pred=np.array([lookup.get((int(b),int(a)),fallback) if swaps else fallback for a,b in pairs])
        return float((pred==pairs.sum(axis=1)%p).mean())
    return {"fallback_class":fallback,"majority_accuracy":evaluate(parts["test"],False),
            "swap_lookup_accuracy":evaluate(parts["test"],True)}
