import numpy as np
import pytest
import torch
from beyond_memorization.data import make_split,validate_split,swapped_mask,lookup_baseline
from beyond_memorization.models import build_model


@pytest.mark.parametrize("p",[5,31,97])
@pytest.mark.parametrize("fraction",[.3,.6])
@pytest.mark.parametrize("seed",[0,1,2])
def test_partitions_exact_and_group_disjoint(p,fraction,seed):
    ordered=make_split(p,fraction,.2,seed,"ordered")
    grouped=make_split(p,fraction,.2,seed,"grouped")
    assert [len(x) for x in ordered.values()]==[len(x) for x in grouped.values()]
    validate_split(grouped,p,True)
    assert not swapped_mask(grouped["test"],grouped["train"]).any()
    again=make_split(p,fraction,.2,seed,"grouped")
    assert all(np.array_equal(grouped[k],again[k]) for k in grouped)


def test_audit_catches_orbit_crossing():
    parts=make_split(5,.6,.2,0,"ordered")
    with pytest.raises(AssertionError):validate_split(parts,5,True)


def test_lookup_is_exact_when_swap_seen():
    parts={"train":np.array([[1,2],[2,2]]),"test":np.array([[2,1]])}
    assert lookup_baseline(parts,5)["swap_lookup_accuracy"]==1.


@pytest.mark.parametrize("name",["mlp","transformer"])
def test_model_backward_and_batch_independence(name):
    torch.manual_seed(0);torch.set_num_threads(1)
    model=build_model(name,5);x=torch.tensor([[1,2],[3,4]])
    logits=model(x)
    assert logits.shape==(2,5)
    torch.testing.assert_close(logits[0],model(x[:1])[0],atol=1e-6,rtol=1e-5)
    loss=torch.nn.functional.cross_entropy(logits,x.sum(-1)%5)
    loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())


def test_unknown_split_rejected():
    with pytest.raises(ValueError):make_split(strategy="oops")
