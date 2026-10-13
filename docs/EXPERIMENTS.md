# Current training and evaluation

The package runs without neighboring checkouts or old checkpoints. `experiments/source_origins.json` identifies the sources and packaging adaptations. The portable controller uses the current large-catalogue recipe for all three datasets; it is not an exact replay of every historical fit. Earlier ERIM comparisons used additional recalibration/symmetric-category variants and dense cross-price baselines. Earlier synthetic confirmation used additional count/price calibration. Do not transfer their numerical results to new runs.

## Models

| Result name | Meaning | Role |
|---|---|---|
| `popularity` | Pooled conditional independent products | Baseline |
| `independent` | Direct shopper/product and store/product tastes, calendar, price, display, advertising | Baseline |
| `mission` | Two individually normalized latent-mission ESP components | Baseline |
| `neural` | Conditional neural autoregression, hidden width 16, rank-8 cross-price bridge | Baseline |
| `size_aware` | Predicted total size followed by personalized weighted subset selection | Baseline |
| `no_price` | Personalized independent model with price frozen at zero | Price ablation |
| `independent_categories` | Category-specific participation and purchase-count curvature, without category links | Structural ablation |
| `basic` | Category counts and relationships between categories | Our model |
| `product` | Rank-4 product links with a bounded mixture of category orders | Our extension |
| `product_fallback` | Converged basic parent retained after rejecting the extension | Selection outcome |

These are concrete implementations, not claims to reproduce every paper's estimator. Benchmark baseline checkpoints load through `experiments.common.load_model` and `experiments.evaluate.Adapter`. The public application CLI loads `basic` and accepted `product` checkpoints.

## Training

1. Estimate pooled preferences on training baskets.
2. Fit personalized preferences with and without price at ridge 1 and 10; select using validation likelihood, including initialization.
3. Freeze preferences and cache mathematically equivalent actual/reference ESP terms.
4. Fit category counts alone, then category parameters alone: the two **warmups**. They use the full training data and the same objective.
5. Fit structural blocks together. Up to ten categories use dense links; larger catalogues use low-rank links. Retain the independent parent if validation worsens.
6. Fit the optional product extension. Accept only a validation improvement with a passing final convergence check; otherwise preserve the failed/unused attempt and retain the basic parent.
7. Fit mission, neural, and size baselines. Failed optional baselines remain explicitly flagged.
8. Freeze selected checkpoints before scoring test outcomes or oracle effects.

The released `basket_simulator.pipeline` also supports small-catalogue `fit_response` and `fit_category_scale`. Those calibration stages are not silently enabled by the common benchmark. Investigate them with an explicit configuration and separate run.

## Stopping and selection

Each attempt has limits of **350 iterations, 500 training objective calls, and 1,800 seconds of timed optimization**. The objective is full-training average negative log likelihood plus ridge. Batch size 512 chunks memory; it is not stochastic training. Setup/final checks sit outside the timer, and one in-progress objective call may overshoot it.

Preference fits use fixed approximate Fisher scaling. Both original and scaled maximum projected gradients must be at most `2e-4`. Other stages use unit scaling. Optimizer success, tiny objective changes, and time limits do not imply passing. This is local active-stage stationarity, not a global optimum or causal certificate.

On failure, reproduce final-state block gradients and inspect objective progress. Allow one final-state continuation only for an improving objective with finite gradients; preserve the original failure. Required failures block dependent stages. Validation-selected and final-state gradient checks are separate. The best checkpoint may precede convergence.

There is no Torch validation-patience stop. The four-hour budget check occurs before optional baseline fits; it is not a whole-pipeline timeout. The memory guard is a lower-bound estimate, not an operating-system RSS cap.

## Evaluation

| Panel | Metrics |
|---|---|
| Predictive fit | Exact likelihood on every test basket; Recall@5/@10, NDCG@5, variance-corrected Brier, 10-bin ECE |
| Basket generation | Product-frequency MAE, full size TV/mean, category-count TV, all category pairs, seeded triples, multiple purchases, household mean-size error |
| Price scenarios | Identical discounts; own, sibling, cross-category, participation and size effects; paired draw errors and training-price-range flags |
| Promotion decisions | One global plan from identical options, under explicitly hypothetical cost fractions 0.3/0.6/0.85 and funding fractions 0/0.1 |
| Scalability | Actual fitting times; five-repeat fresh-process likelihood, generation and scenario timings, parameters and peak RSS |

Generation/recommendation share 4,096 seeded contexts and three independent replicas of 128 draws. Smaller software fixtures use all available contexts. Price scenarios share 32 contexts, one training-popular product per category, eight seeded paired-category discounts and no promotion. The same random seeds pair before/after draws. All models assume the selected catalogue is offered.

Uncertainty uses 1,000 paired household cluster bootstraps, a **separate** store-period sensitivity, and separate Monte Carlo replicas. It conditions on fitted parameters; no retraining uncertainty, joint two-way clustering, or multiplicity adjustment is claimed.

Synthetic evaluation also enumerates all 4,095 nonempty 12-product sets. The independent generator supplies exact generation truth and price responses for 55 scenarios and 64 contexts. Report relative MAE and slope for own, sibling and cross-category effects separately; aggregate checks can hide targeted failures. Promotion regret uses known generator outcomes under hypothetical costs. Oracle truth is never a training input or selection criterion.

For ERIM/Dunnhumby, actual causal response accuracy, achieved shop profit and regret stay **null**. Observational prices and model predictions cannot supply those truths.

## Data

ERIM preparation consumes canonical transactions (`category`, `product_id`, `customer_id`, `store_id`, `basket_id`, `period`) and retail rows (`price_source`, `average_price`, product/store/period keys, display/advertising flags). It selects four training-popular products per training-supported category and uses periods 1–30 / 31–40 / 41–51. The studied contract uses same-period retrospective retail averages, then earlier rows, then training medians. These are not pre-decision shelf offers. The raw builder retains its audited training-cohort policy; that selection is distinct from the later 28-product projection.

Dunnhumby selects 64 training-supported commodity groups and 16 training-popular products per group, using product-ID tie breaks. Checkout keys are `(BASKET_ID, STORE_ID)`. Price covariates use strictly earlier store observations, then strictly earlier national observations, then training references. Current display/mail records are used only within verified coverage. Lagging avoids contemporaneous scored-transaction inputs but does not establish causal identification or true availability.

Both loaders use unknown ID 0 and reject empty, nonbinary or unsupported baskets. Quantities are outside the estimand. Dense NPZ data contain `(N,C,P)` context/member arrays. Dunnhumby uses packed membership and shared store/week contexts. Missing products must not be padded as offered items.

## Using fitted models

The application interface supports `basic` and accepted `product` checkpoints.
Use `basic` if the product extension was rejected. The following examples use an
existing completed run; no retraining is needed.

### Create the context JSON

This exports one observed test context from either dense ERIM/synthetic arrays or
the shared-context Dunnhumby layout. It exports only covariates, never the target
basket. Run from the repository root:

```bash
BASKET_RUN_DIR=runs/erim python - <<'PY'
import json
from pathlib import Path
from experiments.common import Data, OUT

data = Data('test')
batch, _ = data.batch(slice(0, 1))
context = {key: batch[key][0].tolist()
           for key in ('house', 'store', 'week', 'dlp', 'disp', 'mail')}
(OUT / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
PY
```

For your own contexts, use the prepared `data/id_maps.json` for real household and
store IDs; unknown IDs map to 0. Preserve the preparation's calendar encoding.
`dlp[c][p] = log(offered_price / training_reference_price)`; `disp` and `mail` are
the display and advertising covariates. Arrays must have the saved C-by-P shape.
Flattened product index `j = c * P + p` refers to `data/config.json`'s `products`
ordering. `data/reference_price.npy` stores those reference prices. Keep all
selected products offered, as required by the model's current availability law.

### Application commands

For each command below, append these shared arguments:

```text
--checkpoint runs/erim/selected/basic/best.pt
--config runs/erim/selected/basic/config.json
--context runs/erim/context.json
```

| Command | Function and output |
|---|---|
| `python -m basket_simulator demo` | Independent hand-set illustration; needs no checkpoint or context |
| `python -m basket_simulator predict` | Product inclusion probabilities, category participation, expected distinct basket size |
| `python -m basket_simulator generate --draws 100 --seed 7` | 100 nonempty binary C-by-P baskets |
| `python -m basket_simulator recommend` | Product ranking by inclusion probability, before observing the basket |
| `python -m basket_simulator recommend --revealed revealed.json` | Rank additions to a supplied binary C-by-P basket, assuming exactly one missing product; scores are joint log probabilities |
| `python -m basket_simulator what-if --product 0 --factor 0.9` | Before/after predictions and changes for a 10% discount on internal product 0 |

All commands write JSON to stdout; add `--output result.json` to save it. Use
`--mc-draws 8192 --seed 7` to control numerical integration. The basic model
integrates category patterns exactly up to ten categories and samples them above
that size; the product extension uses Monte Carlo prediction. Generation samples
the fitted basket law directly. Larger draw counts reduce simulation error, not
uncertainty about fitted parameters or causal identification.

### Python interface and basket scoring

```python
import json
import numpy as np
from basket_simulator import BasketSimulator, PopulationBasketSimulator

run = 'runs/erim'
model = BasketSimulator.load(f'{run}/selected/basic/best.pt',
                             f'{run}/selected/basic/config.json')
context = json.load(open(f'{run}/context.json'))
prediction = model.predict(context, draws=8192, seed=7)
baskets = model.generate(context, draws=100, seed=7)
log_probability = model.log_probability(context, baskets[0])
ranked_ids, scores = model.recommend(context, draws=8192, seed=7)
discount = model.what_if(context, product=0, factor=0.9, draws=8192, seed=7)

# Explicit unknown shopper: mix known shoppers using training-trip frequencies.
with np.load(f'{run}/data/train.npz') as training:
    population = PopulationBasketSimulator.from_training(
        model, training['house'], representatives=128, seed=7)
unknown_context = {**context, 'house': 0}
population_prediction = population.predict(unknown_context, seed=7)
population_baskets = population.generate(unknown_context, draws=100, seed=7)
```

The population wrapper also supports `recommend`, `what_if`, and
`log_probability`. It supplies an empirical mixture for an explicitly unknown
shopper, conditional on a retained trip; it does not learn a new shopper or
predict shop visits. Basic row-0 inference and this mixture are different choices.

To run the full five-panel comparison for baseline checkpoints, use
`python -m experiments evaluate --run runs/erim`. Baselines are loaded internally
through `experiments.common.load_model` and `experiments.evaluate.Adapter`;
passing their checkpoints to `BasketSimulator.load` is unsupported. Promotion
plan evaluation is in the benchmark, rather than a standalone application CLI.

### Errors and resuming

- A missing `best.pt` means the corresponding selected model is not available;
  inspect `fit_records.json`, `status.json`, and `training.log` first.
- A manifest mismatch means code or inputs changed. Create a new run, or diagnose
  and record an explicit amendment; do not delete the manifest to force reuse.
- An unresolved fit or `stopped.json` requires inspecting its diagnosis before
  any continuation. `monitor` is read-only and never resumes failed stages.
- Invalid context shapes or IDs require fixing the context against the saved
  mappings. Do not pad missing products into the fully offered catalogue.

Use `python -m experiments --help`, `python -m basket_simulator --help`, or
`python -m basket_simulator.pipeline --help` for the complete argument lists.

## Optional SHOPPER

The packaged native myopic SHOPPER adapter uses the upstream sampled loss/Adam and shared observable controls. It is a screening comparator, **not** the paper's Bayesian/thinking-ahead estimator, and remains separate from the common Torch pipeline.

```bash
python -m venv .venv-shopper
.venv-shopper/bin/python -m pip install \
  'git+https://github.com/artefactory/choice-learn.git@d0650088de239b1d4d9926da2a9eb24ae34e0472'
.venv-shopper/bin/python -m experiments.shopper.native --run runs/erim
```

Exact unordered validation is guarded at 12 distinct products. Larger baskets are refused, never dropped. Full-catalogue observed checkouts also require diagnosis because the native negative sampler excludes the observed basket. Native large-catalogue generation comparison remains open; this branch does not claim it was completed.

## Provenance and resume

The protocol precedes preparation; manifests hash prepared data, code before fitting, and selected checkpoints before scoring. An OS lock prevents concurrent training/evaluation writers. Completed fits reuse `selection_record.json`. Incomplete directories are not overwritten. Changed source/input hashes require a new run or a reviewed amendment; the monitor never restarts failures.

The original live research job continues outside this package. These reorganized sources do not change its frozen implementation or relabel any result.
