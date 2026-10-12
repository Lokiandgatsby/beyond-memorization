# Beyond Memorization

**How does evaluation symmetry change what a small neural network appears to generalize?**

Beyond Memorization studies modular addition with an embedding MLP and a small causal Transformer. It compares random splits of ordered inputs with splits that keep operand-swap orbits together, then examines training dynamics, data fraction, and weight decay.

The task is `(a + b) mod 97`: 9,409 input pairs, 97 output classes, and a fully generated dataset. Experiment configurations, logs, checkpoints, and per-example predictions are included.

![Training and validation accuracy](figures/learning_curves.png)

## Quick start

Requires Python 3.10 or newer. The included experiments run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of `source`. If PyTorch is already installed, skip its installation step. GPU experiments can use an existing CUDA-enabled PyTorch installation.

Run inference with an included checkpoint:

```bash
python -m beyond_memorization.predict \
  --checkpoint results/mlp_main_ordered_s0/final.pt 12 35
```

The arithmetic target for this example is 47. Verify the saved predictions, checkpoints, partitions, and extended-run prefixes:

```bash
python -m beyond_memorization.verify
```

## Experimental design

Two partition strategies use exactly the same numbers of training, validation, and test pairs:

- **Ordered:** randomly partition the ordered pairs `(a, b)`.
- **Grouped:** keep `(a, b)` and `(b, a)` in the same partition. Diagonal inputs form singleton orbits.

Both exclude exact duplicate inputs across partitions. Grouped splitting additionally excludes swap-orbit overlap. The protocols test different kinds of held-out input: a new ordered pair versus a new swap orbit.

The fixed matrix contains 24 runs, each with 3,000 optimizer updates. It includes both architectures at a 60% training fraction and Transformer ablations at 30% training data or zero weight decay. Each setting uses seeds 0, 1, and 2. Six exploratory Transformer runs extend the base horizon to 12,000 updates.

Training and validation metrics are recorded every 100 updates. Test predictions are produced at the predetermined final checkpoint. The extended runs reuse the original partitions and are reported separately. Full details are in [the experiment protocol](docs/PROTOCOL.md).

## Results

Test accuracy in percent, reported as mean ± sample standard deviation across three seeds:

| Model / setting | Ordered | Grouped |
| --- | ---: | ---: |
| MLP / 60% train, decay 1 | 98.05 ± 3.37 | 98.57 ± 0.93 |
| Transformer / 60% train, decay 1 | 65.46 ± 20.28 | 84.65 ± 7.79 |
| Transformer / 30% train, decay 1 | 80.85 ± 16.49 | 0.16 ± 0.10 |
| Transformer / 60% train, decay 0 | 38.79 ± 4.79 | 0.46 ± 0.11 |
| Transformer / 60% train, 12,000 updates (exploratory) | 67.76 ± 20.00 | 93.27 ± 2.35 |

The MLP predicts well under both protocols. Transformer performance is sensitive to the training regime; grouping does not consistently make the task harder. With 30% training data or no weight decay, grouped Transformer accuracy is much lower at the fixed horizon.

A control that stores training labels and retrieves the reversed input achieves **59.75%** on ordered base splits and **0.50%** on grouped splits. It illustrates how reversal coverage can support held-out prediction without learning modular addition. It does not establish which mechanism a neural network uses.

No run meets the operational delayed-generalization criterion: training accuracy ≥99%, followed at least 500 updates later by validation accuracy ≥90%. This describes the tested configurations and horizons.

![Architecture and ablation comparison](figures/ablation_comparison.png)

- [Per-run results](results/all_runs.csv)
- [Aggregate results](results/aggregated.csv)
- [Experiment report](docs/Research_Report.pdf) / [Markdown](docs/Research_Report.md)
- [Retrieval control](figures/swap_controls.png)
- [Extended learning curves](figures/extended_dynamics.png)
- [Test error maps](figures/test_error_maps.png)

## Run experiments

Completed run directories are protected against overwriting. Train one configuration into a new output directory:

```bash
python -m beyond_memorization.train \
  --config configs/mlp_main_ordered_s0.json \
  --out reruns/results/mlp_main_ordered_s0
```

Run the full study into a separate root:

```bash
python -m beyond_memorization.suite --execute --root reruns --workers 2
python -m beyond_memorization.extended --root reruns --workers 2
python -m beyond_memorization.analyze --root reruns
python -m beyond_memorization.verify --root reruns
```

The suite creates its root and writes the matrix before executing it. Re-running either family skips completed runs. For a single run, device and horizon are configuration fields; change them before starting an experiment.

The [Colab notebook](colab.ipynb) provides an alternative upload-and-run workflow. It has not been executed in Colab; included results were produced on CPU.

## Rebuild figures and report

```bash
python -m beyond_memorization.analyze
python -m pip install -e ".[report]"
python -m beyond_memorization.report
```

The PDF builder requires DejaVu fonts. Set `BEYOND_MEMORIZATION_FONT_DIR` to a directory containing `DejaVuSerif.ttf`, `DejaVuSans.ttf`, and `DejaVuSans-Bold.ttf` when they are not installed in a standard location. The report uses the complete study, including the exploratory extension.

## Repository layout

```text
src/beyond_memorization/  dataset, models, training, analysis, verification
configs/                 executed configurations and matrix declarations
results/                 logs, split arrays, checkpoints, predictions, tables
figures/                 plots generated from experiment logs
docs/                    experiment protocol and report
tests/                   partition, gradient, and checkpoint checks
colab.ipynb              optional notebook entry point
```

Each run records the seed, hyperparameters, device, environment, split hashes, and final checkpoint. `SHA256SUMS` records release-file checksums. Recorded environment versions describe the original execution environment; requirements specify the supported installation range.

## Interpretation

The comparison uses one modulus and three seeds per setting. Grouping changes orbit coverage and partition composition as well as reversal overlap. Parameter counts differ between architectures. Results describe these joint experimental settings rather than an isolated causal effect of symmetry. The horizon extension is exploratory and reuses the test partitions.

## References

- Power et al. (2022). [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).
- Nanda et al. (2023). [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
- Gromov (2023). [Grokking modular arithmetic](https://arxiv.org/abs/2301.02679).

## License

[MIT](LICENSE)
