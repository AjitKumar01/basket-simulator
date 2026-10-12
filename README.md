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
python -m basket_simulator.pipeline \
  --train data/example/train.npz \
  --validation data/example/validation.npz \
  --config data/example/config.json \
  --out runs/example --seed 0 --maxiter 50 --maxcalls 100
```

The example is for exercising the software. It is not real retail data, and training on it does not establish causal robustness. The output directory must be new; choose a different directory for another run.

The recommended pipeline first fits independent-product preferences, selecting household shrinkage from ridge 1/10. It then freezes those preferences and fits within-category count penalties plus category biases and pair links. The coupled ESP category inclusive value is retained; a matched ablation did not favor removing that coupling. The second stage can retain its parent if extra structure worsens validation. This separates preference learning from count/category calibration.

Training prints one JSON progress record per optimizer iteration. `runs/example/` contains `preferences_0/`, `preferences_1/`, `structure/`, optional `response/` and `category_calibration/`, `selection.json`, and the selected `best.pt`/`config.json`. Each stage contains:

| File | Meaning |
|---|---|
| `best.pt` | Parameters with the highest validation basket log likelihood |
| `final.pt` | Final optimizer parameters |
| `latest.pt` | Most recent completed iteration |
| `config.json` | Model dimensions and settings |
| `fit.json` | Training loss, validation scores, stopping reason and stationarity checks |
| `history.json` | Iteration-level validation scores and timing |
| `protocol.json` | Stage fit settings and selection rule |

The training CLI also records input-data and source hashes in the root `input_manifest.json`.

**A saved checkpoint is not automatically a converged fit.** Check the best and final projected-gradient results in `fit.json`. Predictive selection and optimization stationarity are reported separately. The reported stationarity applies to each stage's active parameters, not a joint optimum. For an explicit single-stage or warm-start fit use `python -m basket_simulator.train` with `--init-checkpoint`; that restores parameter values, not optimizer history. The unrestricted joint refit did not improve the strongest baseline in the repair experiment and is not the recommended fitting route.

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

Optional configuration keys include `order`, `chain_rank`, `preference_mode` (`direct` or `factors`), `category_anchor` (`independent` or `legacy`) and `trainable_parameters`. The generic legacy model also exposes taste/store/hidden/cross ranks. Prefer the staged pipeline for the tested direct-preference, baseline-inclusive coupled model; `train.py` remains available for explicitly selected blocks and legacy fits.

Set `learn_count_offset: true` to learn count mean separately from the penalty on larger counts. For the tested price-calibration recipe, add these two keys to the basic configuration:

```json
"fit_response": true,
"fit_category_scale": true
```

The first stage enables the count offset and fits count/price channels with shopper/store/calendar preferences frozen. The second also learns the existing category inclusive-value scale. Each stage selects on validation likelihood and can retain its parent. Category calibration stops if the response optimizer's final stationarity check fails; inspect its saved history before another run. These stages currently support at most 10 categories. They passed six single/paired price-response checks on an independently seeded synthetic world after the recipe was frozen. That is a stress test within one world family, not certification of causal price predictions on real data.

`memory_budget_mib` optionally rejects a dense-preference fit before allocation when its optimizer-memory estimate exceeds that budget. `memory_estimate.json` records the estimate; covariates and autograd workspace need additional memory. At 3,848 household rows and 8,192 products, the preference tables alone need about 255 MB and ten L-BFGS correction pairs about 5.1 GB. Polynomial computation does not eliminate this storage cost.

A prediction context JSON contains exactly `house`, `store`, `week`, `dlp`, `disp`, `mail`, without a basket target or size. The last three fields are C-by-P lists. Training requires targets; synthetic generation does not.

## Python API

```python
from basket_simulator.pipeline import staged_fit
from basket_simulator.train import load_data
from basket_simulator import BasketSimulator

model, result = staged_fit(config, load_data(train_path, config),
                           load_data(validation_path, config), "runs/custom", seed=0)
simulator = model.reference()  # selected best parameters after fitting
prediction = simulator.predict(context)
baskets = simulator.generate(context, draws=100, seed=7)
```

Or load saved parameters with `BasketSimulator.load(checkpoint_path, config_path)`. The trainable model uses PyTorch automatic differentiation and ESP dynamic programming. Basket likelihood and sampling do not enumerate category patterns or product subsets. Category expectations are exact for up to 10 categories; larger cases use category Monte Carlo with exact within-category expectations and reported standard errors. Price scenarios share category random numbers and report uncertainty in the difference. Use `--mc-draws` and `--seed` to control this integration.

For an unknown shopper, a finite population mixture can average over learned tastes rather than set all shopper effects to zero:

```python
from basket_simulator import PopulationBasketSimulator

population = PopulationBasketSimulator.from_training(
    simulator, training_household_ids, representatives=128, seed=7)
unknown_context = {**context, "house": 0}
prediction = population.predict(unknown_context)
baskets = population.generate(unknown_context, draws=100, seed=7)
```

Use only training IDs/frequencies to build the prior. The mixture supports the same `log_probability`, `recommend` and `what_if` methods. It defines an exact finite mixture; the representative approximation to the broader shopper population is a separate limitation. Save its `households` and `weights` if you need to recreate the same prior. This is an explicit Python API option; the basic CLI's unknown row remains the zero-effect fallback.

Large-category chains use low-rank causal pair links by default (`chain_rank=4` when C>10), and cross-category calculations avoid dense category matrices. With fixed product width/count cap and factor ranks, work grows approximately linearly with category count. The scaling smoke passed through 256 categories/8,192 products. This is a numerical/runtime check, not a fitted full-shop validation. Dense household preferences cost household-count times product-count memory; the generic model also supports factorized preferences when that table is too large. The staged pipeline uses direct preferences and requires R=P to exactly include an independent-product parent.

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
  pipeline.py        preferences first, then count/category corrections
  population.py      finite shopper mixture for unknown shoppers
  model.py           application probabilities, sampling, ranking and scenarios
  example_data.py    procedural example data generator
  __main__.py        application CLI
examples/            reference-context illustration for historical 7x4 checkpoints
tests/               probability, gradient and end-to-end training checks
docs/THEORY.md       complete plain-language theory
pyproject.toml       installation and dependencies
```

No historical research scripts, handoff diaries, raw datasets or fitted checkpoints are included in this model folder. On the reused 28-product ERIM development slice, the repaired staged model improves basket LL from -3.4405 (continued independent parent) to -3.4186, size-distribution TV from 0.0138 to 0.0054, and within-category multiple-product rate from 0.00784 to 0.00277 (observed 0.00260). Recall@5 is essentially level, 0.7082 versus 0.7073. These are a focused warm-start development comparison, not final-test or broad-domain claims. The final structural stage passes its active-block convergence check; predictive checkpoints and a joint optimum remain separately reported. Price effects are observational predictions, not certified causal elasticities. Units, visits, stockouts and dynamics remain outside this model.

A subsequent fit **from scratch** improved ERIM later-period LL from -3.4029 to -3.3833; recall@5 was 0.7034 versus 0.7026. That later period had been inspected by an older model, so it is not an untouched final test. On a whole-shopper holdout, the population mixture corrected the repaired model's predicted mean size from 1.234 to 1.278 (observed 1.281). In a new synthetic development world, count/price calibration reduced size-distribution TV from 0.0684 to 0.0087. The first count/price repair passed five of six price-response checks. Diagnosis traced almost all remaining paired-category error to the individual category responses; fitting the existing category scale then passed all six. With the recipe frozen, a second independently seeded world also passed all six: response errors were 16.8–23.2% of predicting no response, below the fixed 25% threshold. Its generated mean size was 2.379 versus 2.378 observed, and size TV was 0.0139 versus the independent baseline's 0.0521. The new calibration stages have not yet been validated on fresh real data or real causal promotions.
