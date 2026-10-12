# Protocol frozen before the controlled matrix

Experiment design recorded on 9 October 2026.

## Questions
Does withholding complete swap orbits change generalization relative to withholding ordered pairs? Does a smaller training set or removing weight decay change this difference? What can a swap-aware memorizer achieve without learning modular addition?

## Fixed main matrix
24 runs at p=97, seeds 0/1/2. MLP and Transformer each run ordered/grouped splits at train fraction 0.6, validation fraction 0.2 and AdamW weight decay 1. Transformer alone additionally runs train fraction 0.3 and weight decay 0 (one factor changed at a time). The remainder is test. All runs use 3000 optimizer updates, batch size 256 with replacement, width 32, learning rate 0.003, AdamW betas (0.9,0.98), no learning-rate schedule, no dropout, and all parameters subject to decay. Log train/validation every 100 updates. Same seed pairs model initialization across splits. The random ordered split is unrestricted; the grouped split matches exact partition counts and keeps diagonal allocations approximately proportional.

## Evaluation rules
The final step is the evaluation checkpoint. Validation curves diagnose dynamics; no validation-driven stopping or selection is performed. Test labels never enter optimization or checkpoint selection; test accuracy and error strata are computed after the fixed training horizon. Test diagnostics are descriptive post-training analyses of the same predictions. No runs or seeds are discarded. Suite timing includes final evaluation, not installation. Report accuracy as mean and sample standard deviation across three seeds; paired grouped-minus-ordered differences are descriptive, not a definitive statistical test. Baseline lookup and majority use training data only.

## Operational dynamics diagnostic
First logged step with train accuracy >=99% is T_mem; first with validation accuracy >=90% is T_gen. A delay >=500 updates is flagged as a delayed-generalization proxy only. This coarse crossing rule does not establish a sudden transition or mechanism. Missing crossings are right-censored at 3000 updates and are never replaced by zero. A negative result within this horizon does not disprove grokking.

## Interpretation limits
Unrestricted random and orbit-grouped partitions differ in orbit coverage, class counts, and composition, despite matching sample counts. Their difference is an effect of the full splitting procedure, not isolated proof that overlap causes all changes. Ordered models are not intrinsically commutative. Reversing a pair is not exact duplicate contamination under the ordinary ordered-pair evaluation task. No novelty claim is made: this is a controlled empirical study motivated by existing grokking research, not a full reproduction of any published configuration.

## Post-hoc amendment: longer training diagnostic
After the base runs revealed incomplete Transformer training accuracy at step 3000, six additional Transformer base runs (both splits, all three seeds) were declared at 12,000 updates. All other hyperparameters stay fixed and training restarts from the same initialization, so the first 3000 steps should reproduce the base trajectories. This extension is exploratory: some original test outcomes had already been displayed. It is not a new independent test set, and cannot upgrade exploratory observations to confirmatory evidence. No longer-horizon configuration was selected on its test performance; the entire six-setting base family is included. The original fixed-horizon matrix remains separately reported.
