# Basket simulator

Train a model of nonempty shopping baskets, generate synthetic baskets, recommend products, and evaluate price changes. The package includes training code; no checkpoint or neighboring repository is required.

The basic model learns shopper/product preferences, category participation and distinct-product counts. The optional **product-interaction model** also represents relationships between specific products in different categories and uses future offered prices when forming purchase probabilities.

## Install

Python 3.11+; CPU execution:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Train from scratch

Create example data:

```bash
python -m basket_simulator.example_data --out data/example
```

To enable the new model, add these keys to `data/example/config.json`:

```json
"fit_product_interactions": true,
"product_rank": 4
```

Then train:

```bash
python -m basket_simulator.pipeline \
  --train data/example/train.npz \
  --validation data/example/validation.npz \
  --config data/example/config.json \
  --out runs/example --seed 0 \
  --maxiter 350 --maxcalls 400 --time-limit 180
```

Use a new output directory for every run. Example data exercise the software; they do not establish real-world accuracy.

Training first learns personalized preferences, then category/count structure, then the optional product-interaction head. Models and checkpoints are selected using validation basket likelihood. If the new head does not improve validation, the converged original model is retained and unsuccessful head diagnostics are preserved. A proposed improved head must pass final convergence before export. Failed prerequisite convergence stops dependent fitting. Above ten categories, count/category warmups precondition the structural fit before checking its full active gradient.

`runs/example/best.pt` and `config.json` are the selected model. `selection.json` explains the decision. Each stage contains `best.pt`, `final.pt`, `latest.pt`, `fit.json`, `history.json` and `protocol.json`. Progress prints as JSON; the root CLI records input/source hashes. Inspect both **final** and **selected-checkpoint** gradient checks: stage convergence is not a joint optimum or a causal certificate.

## Generate, recommend and ask what-if questions

```bash
python -m basket_simulator generate --checkpoint runs/example/best.pt --config runs/example/config.json --context data/example/context.json --draws 100 --seed 7 --output baskets.json
python -m basket_simulator predict --checkpoint runs/example/best.pt --config runs/example/config.json --context data/example/context.json --mc-draws 8192
python -m basket_simulator recommend --checkpoint runs/example/best.pt --config runs/example/config.json --context data/example/context.json --mc-draws 8192
python -m basket_simulator what-if --checkpoint runs/example/best.pt --config runs/example/config.json --context data/example/context.json --product 0 --factor 0.9 --mc-draws 8192
```

A factor of 0.9 is a 10% discount. Product IDs are zero-based flattened indices. Reference prices remain fixed under price changes. Use `--seed` for reproducibility and `--mc-draws` for integration precision. The new model reports paired whole-basket Monte Carlo errors for what-if differences.

`recommend --revealed revealed.json` scores completion with exactly one missing product. Without `--revealed`, recommendation ranks inclusion probabilities.

## Use your own data

Prepare separate training/validation `.npz` files:

| Array | Shape | Meaning |
|---|---|---|
| `house`, `store` | `(N,)` | Integer internal IDs; row 0 is unknown |
| `week` | `(N,)` | Numeric period |
| `dlp` | `(N,C,P)` | `log(offered_price / fixed_training_reference_price)` |
| `disp`, `mail` | `(N,C,P)` | Display/advertising features |
| `member` | `(N,C,P)` | Binary product membership |

Example configuration:

```json
{
  "C": 3, "P": 4, "R": 4,
  "users": 101, "stores": 11,
  "calendar_support": [1, 30],
  "fit_product_interactions": true,
  "product_rank": 4
}
```

IDs include unknown row 0. Fit product selection, ID mappings and reference prices on training data only. Training supports `1 <= R <= P`: `R=P` includes the ordinary independent-product law, while `R<P` uses an exactly normalized category-capped parent. Unsupported observations are rejected, not dropped. Baskets must be nonempty; categories currently have equal widths and all products are offered. Do not pad missing products as available items. This package consumes prepared arrays, not arbitrary transaction CSVs.

A context JSON contains exactly `house`, `store`, `week`, `dlp`, `disp`, `mail`; the last three are C-by-P arrays. It contains no basket or basket-size target.

Optional `fit_response: true` and `fit_category_scale: true` enable additional count/price calibration, currently limited to ten categories. They can retain their parent; additional calibration was rejected on the studied real-data slice. It is not required to enable product interactions.

## Python API

```python
from basket_simulator import BasketSimulator, PopulationBasketSimulator

simulator = BasketSimulator.load("runs/example/best.pt", "runs/example/config.json")
baskets = simulator.generate(context, draws=100, seed=7)
prediction = simulator.predict(context, draws=8192, seed=7)
change = simulator.what_if(context, product=0, factor=0.9, draws=8192, seed=7)

population = PopulationBasketSimulator.from_training(
    simulator, training_household_ids, representatives=128, seed=7)
unknown_baskets = population.generate({**context, "house": 0}, draws=100, seed=7)
```

The population prior uses training IDs/frequencies only. It mixes whole-basket laws and does not infer a new shopper's individual taste. Save its `households` and `weights` to recreate the prior. Single-stage warm starts are available through `python -m basket_simulator.train --init-checkpoint ...`; optimizer history is not restored.

## Scale and evidence

The product head uses low-rank product interactions, a bounded mixture of category orders, and low-rank category links above ten categories. Training evaluates category prefixes in parallel; generation is sequential and polynomial. Product-head computation and sampling were measured through 256 categories/8,192 products and 4,096 shoppers. These were fixed-work, unfitted measurements, not a converged full-shop fit.

Dense shopper/product preferences still cost shoppers times products. `lbfgs_history: 3` reduces the optimizer-history storage estimate by 70% relative to ten corrections; `memory_budget_mib` rejects a fit whose lower-bound estimate exceeds the budget. Neither removes preference-table or data storage, and autograd needs additional memory.

Two fresh synthetic complement worlds showed cross-category response errors falling from 0.529 to 0.231 and from 0.669 to 0.256. The latter narrowly missed the 0.25 threshold. Mission-world substitution remains imperfect. On the reused 28-product real development holdout, the new head improved likelihood from -3.38330 to -3.38072; recommendation and generation metrics were mixed. There is no universal baseline win, established novelty, or real causal promotion certificate.

This model conditions on an existing trip and models distinct products. It does not yet model quantities, visits, inventory, stockouts or multi-period shop dynamics. Price predictions require suitable offered-price data and independent causal evidence before retailer decisions are treated as validated.

## Code and tests

`trainable.py` contains the basic law; `product_interactions.py` contains the new trainable head and application API; `pipeline.py` selects stages; `train.py` handles fitting/data checks; `model.py` and `__main__.py` expose applications; `population.py` handles unknown-shopper mixtures.

```bash
python -m unittest discover -s tests -v
```

Read [the detailed theory](docs/THEORY.md) for the mathematics, generation procedure, what-if assumptions and evaluation logic. No raw research datasets or checkpoints are bundled.
