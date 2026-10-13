# Basket simulator: training and evaluation

Train the model from scratch, compare baselines, generate baskets, and evaluate price scenarios on **synthetic worlds, ERIM, and Dunnhumby Complete Journey**. This branch contains code and relevant documentation. Data, checkpoints, logs, and historical experiment outputs are excluded.

The model learns shopper/product preferences, category purchase counts, category relationships, and optional product interactions. Read [the theory](docs/THEORY.md) for the mathematics and [the experiment guide](docs/EXPERIMENTS.md) for baselines, data contracts, convergence, and evaluation.

## Install

Python 3.11+, CPU execution. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research,dev]'
```

## Synthetic data

No download is needed:

```bash
python -m experiments prepare --dataset synthetic --family product_ising \
  --seed 942000 --out runs/synthetic-complements
python -u -m experiments train --run runs/synthetic-complements \
  > runs/synthetic-complements/training.log 2>&1
```

Other families: `latent_missions_gumbel` and `latent_missions_gumbel_zero_price`. Defaults: 20,000 training, 5,000 validation, and 5,000 test baskets. The oracle is used only after fitting and validation selection.

## ERIM

Prepare from canonical `transactions.parquet` and `store_week_prices.parquet`:

```bash
python -m experiments prepare --dataset erim \
  --source /path/to/erim/canonical --out runs/erim
python -u -m experiments train --run runs/erim > runs/erim/training.log 2>&1
```

To build canonical tables from the original fixed-width downloads, adjust `configs/erim.json`, place source files under `data/erim/raw`, and supply their SHA256 source manifest:

```bash
python -m experiments.erim.build --config configs/erim.json \
  --output-dir data/erim/canonical
```

The studied projection has seven categories and four products per category. Its prices are retrospective sales averages; these support predictive comparisons, not verified causal effects.

## Dunnhumby Complete Journey

Build transaction/price tables from the original CSVs, or skip the first command if those tables already exist:

```bash
python -m experiments.dunnhumby.build \
  --raw /path/to/complete-journey-csv --out data/dunnhumby/base
python -m experiments prepare --dataset dunnhumby \
  --source data/dunnhumby/base \
  --raw-source /path/to/complete-journey-csv --out runs/dunnhumby
python -u -m experiments train --run runs/dunnhumby \
  > runs/dunnhumby/training.log 2>&1
```

Preparation selects **64 categories × 16 products = 1,024 products** using training data. Splits: weeks 9–82, 83–90, and 91–101. Prices use earlier store observations, then earlier national observations, then a training reference. All nonempty projected baskets are retained; this is a selected catalogue, not the entire shop.

## Results and monitoring

Training automatically starts evaluation after fitting gates. Outputs stay under the chosen run directory:

- `comparisons.html`: one report covering predictive fit, basket generation, price scenarios, hypothetical promotion decisions, and runtime.
- `fit_records.json`: selected models, convergence, failures, and continuations.
- `comparison.json`, `scenarios.json`, `practical.json`: metrics and uncertainty.
- `oracle_comparison.json`: synthetic-only exact generator comparisons and hypothetical promotion regret.
- `selected/<model>/best.pt` and `config.json`: selected checkpoints.

```bash
tail -f runs/dunnhumby/training.log
python -m experiments monitor --run runs/dunnhumby
# Reuse selected checkpoints for evaluation:
python -u -m experiments evaluate --run runs/dunnhumby
```

Use a new output directory for each experiment. Completed stages can be reused with identical source/input hashes. Incomplete stages require diagnosis; they are never silently overwritten. The monitor never restarts training.

## Generate baskets and ask what-if questions

Supply a context JSON containing `house`, `store`, `week`, and C-by-P arrays `dlp`, `disp`, `mail`. `dlp` is log price relative to the fixed training reference; no basket target belongs in the context.

```bash
python -m basket_simulator generate \
  --checkpoint runs/dunnhumby/selected/basic/best.pt \
  --config runs/dunnhumby/selected/basic/config.json \
  --context context.json --draws 100 --seed 7 --output baskets.json
python -m basket_simulator what-if \
  --checkpoint runs/dunnhumby/selected/basic/best.pt \
  --config runs/dunnhumby/selected/basic/config.json \
  --context context.json --product 0 --factor 0.9 --mc-draws 8192
```

`predict` and `recommend` use the same checkpoint/config/context arguments. A factor of 0.9 means a 10% discount. Product indices match the saved catalogue. Use an accepted `selected/product` checkpoint for the extension; if rejected, use `selected/basic`. Baseline checkpoints use the benchmark adapters, not this application CLI.

The original model-only training interface remains available for prepared NPZ files:

```bash
python -m basket_simulator.pipeline --train runs/erim/data/train.npz \
  --validation runs/erim/data/validation.npz --config runs/erim/data/config.json \
  --out runs/erim-model-only --maxiter 350 --maxcalls 400 --time-limit 180
```

Add `fit_product_interactions: true` to a separate configuration copy to enable the extension in the model-only interface. The benchmark fits it automatically.

## Layout and tests

```text
basket_simulator/    model, training, generation, recommendation, what-if
experiments/         benchmark controller, optimizer, baselines, evaluation
  synthetic/         known-truth generators and exact oracle metrics
  erim/              raw readers, canonical builder, selected-basket preparation
  dunnhumby/         transaction/price builder and catalogue preparation
  shopper/           optional native myopic SHOPPER fit for small catalogues
configs/             ERIM source-path configuration
docs/                theory and experiment guide
tests/               mathematical, data-contract, and workflow checks
```

```bash
python -m pytest tests -q
python -m experiments check --out runs/math-check
```

The portable benchmark consolidates current sources using the large-catalogue fitting recipe. It does **not** exactly replay earlier ERIM/synthetic fits that used different calibration stages and baseline variants. Origins and adaptations are recorded in `experiments/source_origins.json`.

The model conditions on an existing trip and predicts distinct products. Real causal price accuracy and shop profit remain unverified. Visits, quantities, stockouts, inventory, and multi-period shop dynamics are outside this model.
