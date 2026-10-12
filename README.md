# Basket simulator

A standalone, trainable model of **nonempty shopping baskets**. It learns which categories appear, how many distinct products are chosen in each, and which products those are. Prices have separate effects on product switching, category participation and distinct-product count.

You can train it from your own basket data, generate synthetic baskets, recommend products and evaluate price scenarios. **No pretrained checkpoint or neighboring research repository is required.**

## Install

Python 3.11 or newer; CPU execution. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Train a first model

Generate a small procedural example dataset and fit from random initialization:

```bash
python -m basket_simulator.example_data --out data/example
python -m basket_simulator.train \
  --train data/example/train.npz \
  --validation data/example/validation.npz \
  --config data/example/config.json \
  --out runs/example --seed 0 --maxiter 50
```

The example is for exercising the software. It is not real retail data, and training on it does not establish causal robustness. The output directory must be new; choose a different directory for another run.

Training prints one JSON progress record per optimizer iteration. `runs/example/` contains:

| File | Meaning |
|---|---|
| `best.pt` | Parameters with the highest validation basket log likelihood |
| `final.pt` | Final optimizer parameters |
| `latest.pt` | Most recent completed iteration |
| `config.json` | Model dimensions and settings |
| `fit.json` | Training loss, validation scores, stopping reason and stationarity checks |
| `history.json` | Iteration-level validation scores and timing |
| `protocol.json`, `input_manifest.json` | Fit settings, data hashes and source hashes |

**A saved checkpoint is not automatically a converged fit.** Check the best and final projected-gradient results in `fit.json`. Predictive selection and optimization stationarity are reported separately. Continue from parameter values with `--init-checkpoint runs/example/best.pt --config runs/example/config.json --out runs/continued`; this is a warm start, not restoration of optimizer history.

## Use the model you just trained

```bash
CHECKPOINT=runs/example/best.pt
CONFIG=runs/example/config.json
CONTEXT=data/example/context.json

python -m basket_simulator predict --checkpoint "$CHECKPOINT" --config "$CONFIG" --context "$CONTEXT"
python -m basket_simulator generate --checkpoint "$CHECKPOINT" --config "$CONFIG" --context "$CONTEXT" --draws 100 --seed 7 --output baskets.json
python -m basket_simulator recommend --checkpoint "$CHECKPOINT" --config "$CONFIG" --context "$CONTEXT"
python -m basket_simulator what-if --checkpoint "$CHECKPOINT" --config "$CONFIG" --context "$CONTEXT" --product 0 --factor 0.9
```

A price factor of 0.9 means a 10% discount. Products use zero-based flattened indices. Scenario calculations hold other inputs and the training reference price fixed. These are conditional model predictions; causal accuracy requires independent evidence.

For completion with **exactly one missing product**, use `recommend --revealed revealed.json`. That file contains a binary category-by-product array of already revealed items. This is not unrestricted next-item prediction.

The inference-only toy illustration also works: `python -m basket_simulator demo`. That separate demo has hand-set parameters; the training example above actually fits a model.

## Train with your own data

Prepare separate training and validation `.npz` files. Each must contain:

| Array | Shape | Meaning |
|---|---|---|
| `house` | `(N,)` | Integer household IDs |
| `store` | `(N,)` | Integer store IDs |
| `week` | `(N,)` | Numeric period index |
| `dlp` | `(N,C,P)` | Log price divided by a fixed training reference |
| `disp` | `(N,C,P)` | Display indicators/features consistent across splits |
| `mail` | `(N,C,P)` | Advertising indicators/features consistent across splits |
| `member` | `(N,C,P)` | Binary observed product membership |

`N` is the number of baskets, `C` the number of categories, `P` the products per category. Every basket must be nonempty and contain at most `R` distinct products per category. Uniform category width and full availability are current implementation assumptions. Do not pad smaller categories with fake available products.

Use a matching JSON configuration:

```json
{
  "C": 3,
  "P": 4,
  "R": 4,
  "users": 101,
  "stores": 11,
  "calendar_support": [1, 30]
}
```

`users` and `stores` include all internal ID rows, including row 0 reserved for unknown IDs. Known IDs should start at 1; map unknown validation IDs to 0. Fit ID mappings, product selection and reference prices on training data only. Preserve them for future inference. `calendar_support` contains the first/last training periods; calendar effects are held at the nearest endpoint outside this range.

For product j, `dlp = log(price_j / reference_price_j)`. A zero is reference price, not a free product. Reference prices must remain fixed under interventions. Choose an appropriate chronological validation split and construct price/promotion features without target leakage. The package consumes prepared arrays; it does not infer prices, availability, category mappings or causal identification from arbitrary transaction CSVs.

Optional configuration keys: `taste_rank` (4), `store_rank` (2), `hidden` (8), `cross_rank` (min(C,4)) and `order` (category index permutation). Defaults match the current research architecture.

A prediction context JSON contains exactly `house`, `store`, `week`, `dlp`, `disp`, `mail`, without a basket target or size. The last three fields are C-by-P lists. Training requires targets; synthetic generation does not.

## Python API

```python
from basket_simulator.trainable import TrainableBasketModel
from basket_simulator.train import fit, load_data
from basket_simulator import BasketSimulator

model = TrainableBasketModel(config, seed=0)
result = fit(model, load_data(train_path, config),
             load_data(validation_path, config), "runs/custom")
simulator = model.reference()  # selected best parameters after fitting
prediction = simulator.predict(context)
baskets = simulator.generate(context, draws=100, seed=7)
```

Or load saved parameters with `BasketSimulator.load(checkpoint_path, config_path)`. The trainable model uses PyTorch automatic differentiation and exact ESP dynamic programming. Application marginals use exact enumeration and currently support at most 10 categories and 10 products per category. Larger models need a scalable inference strategy; this release does not claim whole-store scalability.

## Read the theory and run tests

[Theory in simple words](docs/THEORY.md) explains the full mathematics, what each component learns, what-if assumptions, sampling, recommendation and evaluation logic, plus the measured research evidence and remaining limitations.

```bash
python -m unittest discover -s tests -v
```

## Folder layout

```text
basket_simulator/
  trainable.py       differentiable model and exact ESP
  train.py           data validation, fitting, checkpointing and diagnostics
  model.py           application probabilities, sampling, ranking and scenarios
  example_data.py    procedural example data generator
  __main__.py        application CLI
examples/            reference-context illustration for historical 7x4 checkpoints
tests/               probability, gradient and end-to-end training checks
docs/THEORY.md       complete plain-language theory
pyproject.toml       installation and dependencies
```

No historical research scripts, handoff diaries, raw datasets or fitted checkpoints are included in this model folder. Historical real-data results concern a 28-product ERIM projection: independent products leads recommendation, while the chain improves some basket-generation metrics. Current research fits still fail some convergence checks. Price effects are observational predictions, not certified causal elasticities. Units, visits, stockouts and dynamics remain outside this model.
