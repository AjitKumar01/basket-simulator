# The current basket model, explained from first principles

This guide includes the **legacy separated-price category-chain model** and the
applications already evaluated with its saved checkpoints. It is written so you
can understand the model without reading the research diary. Equations describe
the implemented law, not a proposed future architecture.

The plain-language answer is: **the model decides which categories are present,
how many distinct products to choose from each present category, and which products
those are.** Prices can affect each of these decisions differently. A single
probability law connects all three decisions, so fitting, sampling, recommendations
and price scenarios are mathematically consistent.

The current evidence supports useful conditional basket simulation. It does not
yet establish an accurate causal price simulator or a complete shop digital twin.

## Contents

1. [What a basket means here](#1-what-a-basket-means-here)
2. [Probability and utility](#2-probability-and-utility)
3. [Inputs and learned preferences](#3-inputs-and-learned-preferences)
4. [Within-category choice and ESP](#4-within-category-choice-and-esp)
5. [Three separate roles for price](#5-three-separate-roles-for-price)
6. [The category chain](#6-the-category-chain)
7. [The complete basket probability](#7-the-complete-basket-probability)
8. [How fitting works](#8-how-fitting-works)
9. [Synthetic basket generation](#9-synthetic-basket-generation)
10. [Recommendations](#10-recommendations)
11. [Price and promotion what-if questions](#11-price-and-promotion-what-if-questions)
12. [What each evaluation checks](#12-what-each-evaluation-checks)
13. [What the completed results mean](#13-what-the-completed-results-mean)
14. [From basket simulator to shop twin](#14-from-basket-simulator-to-shop-twin)
15. [Code, inputs and reproducibility](#15-code-inputs-and-reproducibility)

## The repaired model and recommended fitting route

The current recommended pipeline preserves a strong product-preference model and
adds a small, separately fitted basket-structure correction. A matched ablation
showed that removing count/category coupling was not needed for the gain; the
selected model retains the coupled inclusive value.

Its reference preference is

\[
b_j=\lambda_j+H_{hj}+S_{sj}+f(t)^\top\mu_j
    +w_j^D D_j+w_j^M M_j,\qquad u_j=b_j-v_jd_j,\quad v_j\ge0.
\]

Here H and S are direct household/product and store/product effects, with shrinkage
selected on validation. The older four-dimensional taste model remains an option,
but is not the preference representation in the successful repair benchmark.

For the selected coupled model, category appeal is the penalized nonempty subset weight:

\[
I_c=\log\sum_{n=1}^{P}\exp\{\log E_n(u_c)-\rho_c n(n-1)/2\}.
\]

In the recommended structural stage,

\[
\eta_c=I_c+\kappa_c+\sum_{d<c}V_{cd}x_d,
\quad P(x_c=1\mid x_{<c},z)=\sigma(\eta_c).
\]

Within a present category,

\[
A_c(n)=\log E_n(u_c)-\rho_c n(n-1)/2,
\quad P(N_c=n\mid x_c=1,z)=\operatorname{softmax}_n A_c(n),
\]

and a subset of that count has weight exp(sum of its item utilities), normalized
by E_n. The final law is the nonempty-conditioned category chain times these
conditional count/subset laws. Only category intercepts, causal pair coefficients
and rho are fitted in the structural stage; product/price coefficients stay fixed.

This architecture contains the independent-product baseline exactly: at rho=0,
kappa=0 and V=0, category probability is Z_c^+/(1+Z_c^+), with
Z_c^+=sum E_n(u_c), and the conditional subset law is exp(sum u)/Z_c^+.
Multiplying categories and their subsets cancels each Z_c^+, giving the ordinary
independent Bernoulli basket law conditioned nonempty. This identity requires R=P.
It was verified against every saved validation basket, not assumed from matching
marginals. Added structure can therefore start from the baseline and retain it
if validation worsens.

An optional unpenalized anchor uses log(sum E_n(u_c)) for category appeal, so rho
only changes the conditional count law. This option also improved on the baseline,
but a matched control slightly favored the original coupled version in likelihood
and size TV. Therefore uncoupling is not established as the cause of the repair.
The supported changes are preference capacity/shrinkage and fitting structure
separately while retaining the learned nuisance coefficients.

Training is staged: fit independent preferences with ridge selection, freeze them,
then fit structural corrections with the coupled inclusive value. An unrestricted joint refit was tried and dropped
because it worsened validation for the strong baseline. The hierarchy is not a
replacement for sufficient household preference information or sensible shrinkage.

On the ERIM development validation, the repaired model scores -3.4186 versus its
continued independent parent -3.4405 in basket LL. Size TV is 0.0054 versus 0.0138,
and within-category multiple-product rate is 0.00277 versus 0.00784 (real 0.00260).
Recall@5 is 0.7073 versus 0.7082. The LL gain's separate household/store-period
cluster SEs are 0.00172/0.00229. Validation was reused; nuisance parameters were
warm-started; two generation replicas were used. This is promising development
evidence, not a fresh final evaluation. Final structural-block stationarity passed;
selected predictive stationarity and a joint optimum are not thereby certified.

The fuller parameterization described below remains implemented for legacy
checkpoints and explicit experiments. The recommended model is the simpler law
above; it fixes its extra count-price, category-scale, hidden and cross blocks
rather than fitting every supported parameter simultaneously.

### Scaling: what is exact and what is estimated

Training and basket likelihood were already polynomial. The old application layer
introduced exponential work by listing all category patterns and product subsets.
The revised application implementation uses ESP prefix/suffix dynamic programs and
samples categories sequentially, conditioned on eventual nonemptiness.

When no earlier category has appeared, let q_c^0 be the category's probability on
the all-absent prefix, and Q_c the probability that at least one category appears
from c onward on that prefix. The conditioned first-presence probability is
q_c^0/Q_c. Q_c is computed backwards in log space by
Q_c=q_c^0+(1-q_c^0)Q_{c+1}. After a category appears, use the ordinary chain
probabilities. This is exact and requires no rejection, even when nonempty baskets
are extremely rare. Within-category counts and subsets are sampled with ESP
suffix probabilities; product subsets are never listed.

For larger category counts, causal pair links use
V_cd=a_c^T b_d for d<c. Their contribution is a_c^T sum_{d<c} b_d x_d,
computed through a running prefix. Cross-category context effects are also
multiplied as low-rank factors without constructing a dense C-by-C matrix.
At fixed ranks, product width and count cap, this work is linear in C. Dense links
remain a compatibility option for small catalogues.

Exact category marginalization of a general nonlinear chain still needs a sum over
patterns. Small C<=10 uses that sum. Large C uses exact ancestral category draws
and exact conditional product expectations; this is Rao-Blackwellized Monte Carlo,
with reported standard errors. Price differences use common random category draws
and report paired difference errors. Exact sampling is not the same as exact
marginal expectation. Increasing integration draws reduces Monte Carlo error,
not parameter uncertainty or causal confounding.

Numerical scaling checks reached 256 categories and 8,192 products (32 per category),
without category enumeration or a dense causal category matrix. For 256 categories,
256 whole-basket samples took about 0.086 seconds and a 4,096-draw category-marginal
estimate about 0.050 seconds on the local machine. These hand-initialized runtime
checks do not establish predictive or causal performance on a large real catalogue.
Direct household tables still cost O(households times products); uniform category
width and full availability remain assumptions. R=P in the recommended exact-
independent-parent pipeline can be more expensive than a small count cap.

## 1. What a basket means here

Suppose a shop sells several brands of margarine, tissue and tuna. A basket could
contain one margarine and one tissue. The model predicts a probability for that
whole set, rather than only predicting an independent score for each item.

For the current real-data exercise, the catalogue is a deliberately restricted
slice of the ERIM retail panel: four training-selected products in each of seven
categories, giving 28 products. A recorded shop basket is projected onto those
products. Baskets with no selected products are outside the fitted population.
The categories are brownie, ketchup, margarine, peanut butter, sugar, tissue and tuna.

Thus a modeled basket of size one can come from a much larger actual shopping trip.
The observed mean size of about 1.29 is the number of **selected distinct products**,
not the number of products bought in the entire shop. Buying two packs of the same
product still contributes one selected product. This distinction matters for revenue,
stock planning and comparisons with earlier synthetic experiments.

The question modeled is:

> Given this household, store, period and commercial context, what nonempty set of
> these selected products would appear in a basket?

It does not answer how often the person visits, whether they purchase nothing, how
many units they buy, or what their next visit looks like.

Notation used below:

| Symbol | Meaning |
|---|---|
| \(z\) | Context: household, store, period, prices and promotions |
| \(c\), \(j\) | Category and product indices |
| \(C\), \(P\) | Number of categories and products per category; currently 7 and 4 |
| \(S\), \(T_c\) | Whole basket and its subset in category \(c\) |
| \(x_c\) | 1 if category \(c\) appears, otherwise 0 |
| \(N_c\) | Number of distinct products in category \(c\) |
| \(R\) | Maximum supported distinct count in one category; currently 4 |
| \(b_j\), \(u_j\) | Reference-price preference and actual-price utility |
| \(d_j\) | Log price relative to a fixed training reference |

We use \(z\) for context to avoid confusing it with the binary category pattern \(x\).

## 2. Probability and utility

A probability is a number between zero and one. All mutually exclusive possibilities
must sum to one. The model first assigns a score, called utility, and then converts
scores into probabilities.

For a single decision between products, utility \(u_j\) gives positive weight
\(w_j=\exp(u_j)\). Dividing each weight by their sum produces probabilities.
Utilities themselves can be negative; their exponentials cannot.

For a yes/no category decision, the conversion is the sigmoid:

\[
\sigma(a)=\frac{1}{1+\exp(-a)}.
\]

Score zero means probability one half. A score of about 1.10 means probability
three quarters. A score of about −1.10 means probability one quarter. These are
decision scores, not money or directly measured satisfaction.

Calculations use log probabilities because multiplying many small probabilities
can lose numerical precision. The function
\(\operatorname{logsumexp}(a)=\log\sum_i\exp(a_i)\) computes a log normalizer stably.
Exact normalization means the probabilities sum to one according to the model's
mathematics. It does not mean the model's assumptions match shoppers perfectly.

## 3. Inputs and learned preferences

Each context contains a household ID, store ID, period index, relative log prices,
display indicators and advertising indicators. These must use the same mapping and
preprocessing as training. For price:

\[
d_j=\log(p_j/p_j^{\rm ref}).
\]

The reference \(p_j^{\rm ref}\) is determined from training data and stays fixed.
At reference price, \(d_j=0\). A 10% discount adds \(\log(0.9)\approx-0.10536\)
to the feature. A 10% increase adds \(\log(1.1)\approx0.09531\). These changes
are not equal in magnitude, even though both are called 10% changes.

The reference-price baseline for product \(j\) is

\[
b_j(z)=\lambda_j+\theta_h^\top\alpha_j
       +\xi_s^\top\zeta_j+f(t)^\top\mu_j
       +w^{\rm display}_j D_j+w^{\rm advert}_j M_j.
\]

Here \(\lambda_j\) is overall product appeal. Household vector \(\theta_h\)
and product vector \(\alpha_j\) summarize household preferences with four learned
coordinates. Store vector \(\xi_s\) and product vector \(\zeta_j\) use two
coordinates to capture store differences. Their inner products mean that a household
or store need not receive a separate unconstrained parameter for every product.

The calendar feature is

\[
f(t)=(\sin\phi,\cos\phi,\sin2\phi,\cos2\phi),\qquad
\phi=\frac{2\pi}{52}\{\operatorname{clip}(t,1,30)-1\}.
\]

The current calendar effects stop changing after training period 30 and before period
1. This is an explicit conservative extrapolation rule. It is not evidence that
future seasonal effects have been learned. Prices and other scenario inputs can
still change outside that interval.

Actual-price product utility is

\[
u_j(z)=b_j(z)-v_j d_j,\qquad v_j\ge0.
\]

A higher price therefore lowers that product's direct utility. The broader research
class supports a household multiplier \(\exp(\gamma_h)\), but **\(\gamma_h=0\)
is fixed in the current fits**. The current model learns household tastes, not an
active household-specific direct price slope. Stored legacy parameter names do not
prove that those extensions were fitted.

The evaluated law assumes all selected products are offered. Display and advertising
can alter appeal through their own coefficients; they are not interchangeable with a
price reduction. Retail price observations are derived from the available retail
feed, with previous/reference-price fallback where needed. A fallback is a feature
construction assumption, not evidence that the shopper faced that exact price.

## 4. Within-category choice and ESP

Within a present category, the model first chooses a distinct count \(n\), then
chooses an unordered subset of exactly \(n\) products. It never picks the same
product twice in that subset.

For weights \(w_j=\exp(u_j)\), the elementary symmetric polynomial is

\[
E_n(u)=\sum_{T:\,|T|=n}\exp\!\left(\sum_{j\in T}u_j\right).
\]

In plain words: list every subset with the required size, multiply the item weights
within each subset, then add those products. This is the total weight for that size.
It normalizes product choice:

\[
P(T_c\mid N_c=n,z)=
\frac{\exp(\sum_{j\in T_c}u_j)}{E_n(u_c)}.
\]

For example, three products A, B, C with weights 2, 1, 1 give
\(E_1=4\), \(E_2=2\cdot1+2\cdot1+1\cdot1=5\), and \(E_3=2\).
If exactly two are selected, AB and AC each have probability 2/5; BC has probability
1/5. If A becomes cheaper and its weight rises to 3, these probabilities become
3/7, 3/7 and 1/7. The count remains two in this conditional calculation; only
composition changes.

The research implementation avoids explicitly listing every subset. The recurrence

\[
E_n(w_1,\ldots,w_k)=E_n(w_1,\ldots,w_{k-1})
                   +w_kE_{n-1}(w_1,\ldots,w_{k-1})
\]

starts from \(E_0=1\) and impossible positive counts having weight zero. It costs
roughly products times supported counts, rather than all \(2^P\) subsets, and is
implemented in log space. The current application code also uses ESP prefix/suffix dynamic programming.
Exhaustive subset tables remain only as a deliberately bounded tiny-model diagnostic.

Multiple products from the same category receive the size penalty

\[
q_c(n)=\rho_c\frac{n(n-1)}2.
\]

Each additional within-category pair contributes \(\rho_c\). Positive values
discourage buying multiple substitutes. A count of one has no pair penalty.
The fitting bounds allow negative \(\rho_c\), which would favor multiple products;
the current saved fits have positive values. This is a category-level multiplicity
effect, not an arbitrary learned interaction for every product pair.

## 5. Three separate roles for price

One price coefficient cannot reliably decide all of brand switching, category
participation and extra distinct products. The current architecture separates them.

### Product switching

The \(v_j\) coefficients in \(u_j=b_j-v_jd_j\) decide which products are more
attractive at a given count. Lowering the price of one brand can shift probability
away from the other brands. The ESP denominator ensures their shares adjust together.

### Distinct count within a category

Define the reference and actual-price log size weights:

\[
A_c^0(n)=\log E_n(b_c)-q_c(n),\qquad
A_c^u(n)=\log E_n(u_c)-q_c(n).
\]

The count score is

\[
A_c(n,z)=A_c^0(n)+s_c\{A_c^u(n)-A_c^0(n)\},\qquad s_c\ge0,
\]

and

\[
P(N_c=n\mid x_c=1,z)=
\frac{\exp A_c(n,z)}{\sum_{r=1}^{R}\exp A_c(r,z)}.
\]

At \(s_c=0\), prices still change brand selection, but do not directly change
this category's conditional count distribution. At \(s_c=1\), the ordinary
utility-based size law is recovered. Larger values strengthen its price response.
This is a distinct-product count, not a number of packs. Several fitted scales are
small or zero, so the model can express mainly switching without forcing large
count expansion.

### Category participation

At reference price the inclusive score is

\[
I_c^0(z)=\operatorname{logsumexp}_{n=1}^{R} A_c^0(n).
\]

It summarizes how attractive the category's possible nonempty subsets are. In the
current separated model:

\[
I_c(z)=I_c^0(z)-\sum_{j\in c} a_{cj}d_j,\qquad a_{cj}\ge0.
\]

These category price loadings \(a_{cj}\) are learned separately from the product
switching slopes \(v_j\). They allow a discount to produce a strong brand switch
but little increase in category participation, or the reverse.

In the previous coupled variant, the score instead was
\(I_c=\operatorname{logsumexp}_n A_c^u(n)\). The same utilities then controlled
product switching and category response. Separating these mechanisms was an
architectural repair. It is not itself proof that the new fit is causally correct
or better on every metric.

The three components are separately parameterized but jointly fitted through the
whole-basket likelihood. Nonprice baseline and promotion changes affect the reference
scores as well. There is no extra price regression attached after basket fitting.

## 6. The category chain

The model visits categories in a fixed computational order and makes a yes/no decision
for each. This order is a probability factorization, not an observed shopping sequence.
The current real-data runs use catalogue order. Earlier experiments also tested reversed
order; exact normalization alone does not guarantee order-invariant fitted predictions.

First construct a context-dependent base score:

\[
B_c=\kappa_c+\operatorname{softplus}(\tau_c)I_c
       +\sum_{d\ne c}K_{cd}I_d+\theta_h^\top U_c,
\qquad K=PQ^\top,\quad K_{cc}=0.
\]

Softplus is \(\log(1+\exp(t))\), a positive transformation. Own-category appeal
enters positively. The low-rank matrix \(K\) allows other categories' appeal to
affect this category. Its signs are unrestricted. Shared household preferences
also affect category presence.

For clarity, index categories in their chosen chain order. Let \(m_c\) be the
number selected before category \(c\). Its decision score is

\[
\eta_c=B_c+\sum_{d<c}V_{cd}x_d
       +\omega_c^\top\operatorname{ReLU}\!\left(h_0+\sum_{d<c}W_dx_d\right)
       +\psi_0[m_c]+\psi_{1c}m_c.
\]

Here \(\operatorname{ReLU}(r)=\max(0,r)\). The \(V\) terms express direct
links between earlier and later category selections. The hidden layer, currently
eight coordinates, captures combinations of earlier selections. The count terms
\(\psi_0,\psi_1\) are supported by the original core but **fixed to zero in the
current fit**. This whole score becomes a probability through sigmoid:

\[
P(x_c=1\mid x_{<c},z)=\sigma(\eta_c).
\]

Multiplying these conditional decisions gives a normalized distribution over every
binary category pattern:

\[
P_{\rm chain}(x\mid z)=
\prod_c\sigma(\eta_c)^{x_c}\{1-\sigma(\eta_c)\}^{1-x_c}.
\]

There are \(2^7=128\) patterns in the historical case, so exact enumeration for
category marginals is inexpensive. The chain law is normalized for any category
count, but this particular exact marginal enumeration grows exponentially with \(C\).
Large catalogues need a different integration strategy or a different factorization.

Positive category links can produce co-purchase dependence; negative links can
produce competition. These are fitted associations within this law. A positive link
does not automatically establish causal complementarity.

## 7. The complete basket probability

The data contain nonempty projected baskets. The model therefore removes the all-zero
category pattern and renormalizes:

\[
P_+(x\mid z)=\frac{P_{\rm chain}(x\mid z)}{1-P_{\rm chain}(0\mid z)},
\qquad x\ne0.
\]

This conditioning matters. If two otherwise independent categories are each selected
with probability one half, their four patterns each initially have probability one
quarter. Removing the empty pattern gives probabilities one third for A only,
B only and A+B. Each category now appears with probability two thirds, and the pair
probability is one third, rather than \((2/3)^2\). Conditioning itself creates
dependence; observed dependence need not come solely from explicit chain links.

The final law for a nonempty basket is

\[
\boxed{
P(S\mid z,S\ne\varnothing)=P_+(x(S)\mid z)
\prod_{c:x_c=1}
\left[
 P(N_c=|T_c|\mid x_c=1,z)
 \frac{\exp(\sum_{j\in T_c}u_j)}{E_{|T_c|}(u_c)}
\right].
}
\]

Every category pattern has a probability, every present category's count probabilities
sum to one, and each fixed-count subset distribution sums to one. That establishes
normalization of the complete law. Different present categories' product/count choices
are independent **conditional on their category pattern and context** in this model;
category presence is not generally independent.

One structural limitation remains: the price of a product in category A can affect
category B's presence, but does not directly change B's conditional brand shares or
within-category count. Those use B's utilities. The cross-category response into B
therefore has a restricted shape. The model does not contain arbitrary product-to-product
cross-category substitution. This is an architectural scope limit, not a proven cause
of every failed empirical metric.

The model supports all 15 nonempty within-category subsets when \(P=R=4\).
If a different run uses \(R<P\), larger within-category sets have probability zero.
Do not compare likelihoods on different basket supports without checking this.

## Count mean, unknown shoppers and the latest checks

The optional count intercept \(\nu_c\) changes each count weight to

\[
 A_{c,n}=\log E_n(\exp u_c)+\nu_c(n-1)
          -\rho_c n(n-1)/2.
\]

Increasing \(\nu_c\) encourages additional distinct products. Increasing
\(\rho_c\) suppresses the upper tail more strongly. Learning both separates count
mean from curvature; setting \(\nu_c=0\) exactly recovers the earlier law. The
same modification enters reference count weights and the coupled category inclusive
value. It is an unpenalized count intercept, like the existing count/category
intercepts. `learn_count_offset` enables it. The optional `fit_response` stage
keeps shopper/store/calendar preferences fixed while fitting count and price
channels. Its small-category cross-link reparameterization preserves the initial
probability law; it is currently limited to at most ten categories. Then
`fit_category_scale` adds the existing category scale \(\operatorname{softplus}(\tau_c)\)
to that same response block. This lets a category translate its inclusive value
into participation with a learned sensitivity rather than a fixed sensitivity of
one. Product tastes remain fixed. This stage requires the response optimizer's
final active-block stationarity check to pass. Selection still uses observed
validation basket likelihood, with the parent available.

For an unknown shopper, averaging preferences first is generally wrong because
probabilities are nonlinear functions of those preferences. The explicit
`PopulationBasketSimulator` instead defines

\[
 p_{\rm new}(S\mid z)=\sum_{h\in\mathcal H}w_h p(S\mid h,z),
 \qquad w_h>0,\quad\sum_h w_h=1.
\]

Its prior uses only training trip frequencies and a fixed representative sample.
Likelihood is a log-sum-exp over components; expected incidence is their weighted
average. Generation draws one shopper component, then the complete basket. Thus it
preserves dependencies within a basket. Completion scores candidate baskets under
the mixture. Price scenarios keep prior weights fixed; changing who visits the
shop remains outside this conditional model. This prior can improve aggregate
calibration without learning the unknown individual's tastes.

The later ERIM comparison fitted the released basic recipe from scratch and froze
it before scoring: LL improved from -3.4029 to -3.3833, with nearly level recall.
These targets had been inspected by an older model. A deliberate whole-shopper
holdout exposed zero-effect size bias: repaired mean1.234 vs real1.281. The finite
population mixture predicts1.278; sampled size TV drops from .0326 to .0096.

The initial new synthetic world exposed a count-shape failure: size TV .0684 and
predicted mean2.283 vs observed2.365. The validation-selected count/price followup
gives TV .0087 and mean2.371. It initially passes five of six price-response
checks. Paired discount category-response error is29.4% of the zero-response error,
exceeding the unchanged25% target. An exact decomposition attributes about99% of
that error to constituent single-action errors adding up; the nonadditive
interaction contributes about1%. Allowing the existing category scale to learn
reduces that failed error to24.3%, passing all six checks. This world is development
evidence because its errors informed the repair; earlier failures are preserved.

The complete recipe was then frozen before fitting a second world with independent
parameters and sample seeds. It uses only observed baskets and prices for training
and validation selection; oracle response truth is consulted after selection.

| Fresh-world check | Independent parent | Calibrated chain |
|---|---:|---:|
| Basket LL | -5.6081 | -5.2514 |
| Recall@5 | .5986 | .5990 |
| Generated size TV | .0521 | .0139 |
| Within-category multiple rate | .1676 | .2556 |

The true multiple rate is .2570. Predicted mean size is2.3792 versus observed2.3783.
All six single/paired response checks pass: relative mean absolute errors are
16.8–23.2% of the error from predicting zero response, with slopes .91–1.16.
The fixed pass rule requires relative error at most25% and slope between .75 and1.25.
The independent parent fails all six. This supports the repair within one stress-world
family; broader synthetic families and real causal promotion accuracy remain open.
All final stages pass their local gradient checks. Some validation-selected best
checkpoints fail those checks, and no joint optimum is claimed. The new response
stages were subsequently tried on the existing real development slice. They did
not improve it, as detailed below; fresh real retail and causal evaluation remain open.

### Real transfer and campaign audit

The full response block was warm-started from the from-scratch basic ERIM model.
All 250 validation iterates were worse than the parent, so predictive selection
kept that parent. The final projected gradient was .00377, above .0002; category
calibration and dependent scoring were stopped. The biggest gradient was in the
cross-category price block. Training has only three three-product category events
and no four-product events, so a separately learned count mean and curvature have
much weaker support here than in the synthetic world. These are measured diagnostics;
they do not establish a unique explanation for every real price error.

A smaller, separately recorded fit froze cross-category links and count parameters,
allowing only product switching, direct/inclusive-value category price channels,
category bias and category scale to learn. It converged, but every validation
iterate still lost to its parent. This distinguishes an optimization failure from
a predictive failure: more optimizer iterations cannot make the already converged
restricted fit improve its validation score. The extra calibration was rejected.

For a fairer comparison, the independent baseline also recalibrated its product
price slopes with household/store/calendar preferences fixed. Validation selected
that baseline before scoring the reused later period.

| Real later-period metric | Recalibrated independent | Retained basic chain |
|---|---:|---:|
| Basket LL | -3.3955 | -3.3833 |
| Recall@5 | .7039 | .7026 |
| Size TV | .0113 | .0148 |
| Within-category multiple rate | .00619 | .00220 |

The observed multiple rate is .00203. LL gain is .01221, with household-cluster
SE .00185 and store-period-cluster SE .00273. Both restricted final fits pass their
active-stage stationarity check; their selected best checkpoints fail it separately.
This is repeatedly inspected development data, not a fresh causal or final test.

The campaign adapter covers all 2,975 assigned households and all four Cottonelle
products. Original documentation distinguishes three sources: actual promoted
shelf prices, weekly aggregate pre-coupon sales averages, and weekly household
shopping summaries. Only the first gives a shelf offer, and only for promotion
rows. In the campaign it covers about40.5% of tissue purchase records; no store-week
has offers for all four Cottonelle products. Trip-specific shopping records begin
at the campaign's final week. The weekly summaries cover earlier activity but
contain no store/day/trip. Annual flags indicate card usage, not complete panel
participation. A missing purchase is therefore not always verified non-purchase.

Pooled coupon-arm households recorded .435 more Cottonelle units per assigned
household than controls; a within-cell household sampling interval is [.089,.787].
The pre-campaign contrast was already .096. A conservative five-triplet cell
sign-flip sensitivity gives p=.0625; this is also its minimum attainable two-sided
p. Neither calculation proves the undocumented randomization unit. These are
recorded arm associations, not a pure-price verdict. Conditioning on campaign
nonempty baskets can also select shoppers after treatment. Coupon receipt,
attention and redemption cannot simply be replaced by the realized average
redeemed discount. An exposure/visit and quantity layer plus suitable treatment
data are needed before testing shop-wide promotion margin.

Category computation now scales polynomially, but dense preferences still require
households times products storage. Ten optimizer history pairs cost about20times
the parameter bytes. `memory_budget_mib` can reject an oversized dense fit early;
it does not turn dense training into a sparse or validated factorized full-shop fit.

## 8. How fitting works

For each observed basket the model evaluates the complete probability above. Training
chooses parameters that make the recorded baskets probable. Taking negative log
probabilities turns this into a minimization problem:

\[
L(\vartheta)=-\frac1{n_{\rm train}}\sum_i\log P_\vartheta(S_i\mid z_i)
 +\frac{\lambda_{\rm ridge}}{2n_{\rm train}}
   \sum_{k\in\mathcal A}\vartheta_k^2.
\]

The current chain uses ridge strength 10. Active nonintercept parameters are penalized;
the product intercepts \(\lambda_j\), category intercepts \(\kappa_c\) and
\(\rho_c\) and optional count intercepts \(\nu_c\) are excluded from this ridge term. Inactive legacy parameters are not
new sources of learned behavior. Full-batch calculations are chunked for memory,
and L-BFGS-B optimizes with bounds, including nonnegative physical price slopes
and count scales. Some later runners use fixed parameter-block scaling to help
optimization; that changes optimizer coordinates, not the probability law.

Training and validation serve different purposes. Training updates parameters.
Validation chooses the predictive checkpoint. An optimizer can keep reducing the
penalized training objective while making future basket predictions worse. Conversely,
the best predictive checkpoint can be selected before the training optimum is reached.

The stationarity check asks whether any allowed parameter direction still has a
substantial local training gradient. At a lower bound, a gradient suggesting movement
below that bound is ignored. The maximum remaining projected gradient must be at most
\(2\times10^{-4}\) under the recorded checks. An optimizer message saying it stopped
does not replace this check.

The historical selected chain checkpoints fail that convergence criterion. In the
repair experiment, the final independent preference fit and final structural fit
pass for their respective active parameters. Their validation-selected checkpoints
do not pass. This is stage-specific stationarity, not certification of a joint
optimum. SHOPPER stationarity remains uncertified.
This does not invalidate the numerical inference code or erase measured predictive
scores. It does limit claims that differences reflect well-optimized model classes,
and leaves parameter-based price responses less settled.

Latent taste factors have scale/rotation ambiguities and correlated parameters can
produce similar likelihoods. Price, advertising, calendar and store effects can also
compete to explain the same observational variation. Bounds, regularization, sensible
coordinates and multiple starts help, but do not create causal price identification.

## 9. Synthetic basket generation

Sampling means drawing a random basket according to the fitted probabilities, rather
than choosing the single most likely basket or copying an observed basket.

For a fixed context:

1. Compute the conditional nonempty category-pattern probabilities and sample one.
2. For each present category, sample a count from its count distribution.
3. Sample a distinct subset of that size using its utility-based ESP distribution.
4. Join the category subsets to obtain the whole basket.

The current adapter implements these steps directly through a conditioned chain
and ESP suffix tables. It does not build a table of every possible product subset.

The research ESP method can sample a subset without enumerating it. For a remaining
pool \(A\), when exactly \(k\) items remain to be picked, the probability of including
the next item \(i\) is

\[
\frac{w_i E_{k-1}(A\setminus\{i\})}{E_k(A)}.
\]

Include or skip the item accordingly, update the required count, and continue.
This respects both the count and product weights.

The input does not include the real basket size or its purchased products. The
application API rejects extra context fields, preventing accidental use of targets.
Random seeds make samples repeatable; two seeds will usually produce different
individual baskets from the same underlying probabilities.

To simulate a customer mix, supply contexts from that mix and draw baskets. The
latest evaluation reuses real validation contexts but samples their item sets and
sizes independently of targets. These are synthetic baskets conditional on real
covariates, not fully synthetic people or a privacy guarantee. Reusing household
identities and commercial contexts can preserve sensitive information.

The expected inclusion of product \(j\) in category \(c\) is

\[
\pi_j(z)=P_+(x_c=1\mid z)
\sum_{n=1}^{R}P(N_c=n\mid x_c=1,z)
P(j\in T_c\mid N_c=n,z).
\]

Within an ESP distribution, the last probability is
\(\exp(u_j)E_{n-1}(u_{c\setminus j})/E_n(u_c)\). Expected distinct size is
\(\sum_j\pi_j\). Exact expectations do not require a large synthetic sample.
Sampling is useful when the application needs individual basket realizations,
distributional shapes or a downstream simulation consuming baskets.

Matching just mean size is insufficient. A generator could average one and three
items correctly while never generating the common two-item baskets. It also needs
correct item frequencies, category combinations, within-category multiplicity and
behavior across household/store groups.

## 10. Recommendations

Before seeing any selected products, rank products by their inclusion probabilities
\(\pi_j(z)\). That answers which products are likely to appear in this context.
It is not a prediction of click-through or a causal estimate of displaying a recommendation.

For the evaluated completion task, reveal the basket except for **exactly one** missing
product. For every eligible candidate \(j\), score

\[
\operatorname{score}(j)=\log P(R_{\rm revealed}\cup\{j\}\mid z).
\]

Conditioning on the revealed set, a known final size, and a uniform one-item hiding
mechanism contributes the same factor for every candidate. Ranking by joint basket
probability is therefore appropriate for this task. Probabilities over candidates
can be obtained by normalizing their scores with logsumexp. Purchased products
already revealed are excluded.

This is not the same as a generic next-item query when any number of products could
still be added. For that query the model would need to sum over all possible final
baskets consistent with the revealed set, or introduce a purchase-order model.
The present unordered dataset does not supply an observed shopping sequence.

Our current marginal recommendations lose to the independent-product baseline.
Generation quality can still be better, because predicting likely individual items
and reproducing their joint combinations are different tasks.

## 11. Price and promotion what-if questions

### The calculation

For a price scenario, start with the same context and fitted parameters. If product
\(j\) receives multiplicative price factor \(r\), replace

\[
d'_j=d_j+\log r.
\]

Keep all other prices and context inputs fixed unless the scenario explicitly changes
them. Do not recompute the reference price. Recompute utilities, count scores,
category scores, chain probabilities and final marginal probabilities from the same law.

Report changes such as

\[
\Delta\pi_j=\pi_j(z')-\pi_j(z),\qquad
\Delta\operatorname{size}=\sum_j\Delta\pi_j.
\]

For a promoted product, separate its own inclusion change, the sum for sibling brands,
the sum for other categories, and the net distinct-size change. This prevents a large
brand switch from being presented as a large expansion in total demand.

The current 7-by-4 model computes these expectations exactly by category enumeration
and within-category probabilities. Drawing synthetic baskets before and after is
optional and introduces additional Monte Carlo noise. SHOPPER scenarios in the saved
comparison use a separate verified importance-integration method; its integration
diagnostics are not price-accuracy diagnostics.

### A concrete saved-model example

For margarine product `marg:0029000008220`, the completed supported 10% discount
scenario over 220 contexts predicts, averaged over the two tuned checkpoints:

| Quantity | Change per conditional projected basket |
|---|---:|
| Discounted product inclusion | +0.01442, about +1.44 percentage points |
| Other margarine products, summed | −0.01175 |
| Other categories, summed | −0.00221 |
| Expected distinct basket size | +0.00046 |

Most predicted gain is switching. The net is small because positive and negative
terms nearly cancel. Independent products predicts a larger net increase, about
0.00290, while SHOPPER predicts a smaller own inclusion change, about 0.00136.
No actual randomized intervention outcome is available to establish which is correct.
The example was chosen by training frequency, not by selecting the largest predicted gain.

### Why price helps, and why it is insufficient

Without price features, two otherwise identical contexts have the same predicted
distribution when a price changes. Price features let the model express economically
meaningful differences and can improve historical basket prediction. The separated
roles avoid forcing brand switching and category/count expansion to move together.

However, historical shops may discount products precisely when demand is expected
to change. Displays, mail advertising, competitors, stock and latent customer needs
may move with price. A fitted conditional relationship can then mix price effects
with those other causes. Good likelihood does not disentangle them automatically.

Real causal what-if validity needs either credible identifying assumptions with
adequate variation and controls, or independent intervention evidence. Randomized
promotions are especially informative. Synthetic worlds provide known intervention
truth and help diagnose whether the architecture can learn behavior, but passing
them would not remove confounding from real retail data.

Nonnegative direct price coefficients enforce local direct utility direction. They
do not ensure every whole-basket response is monotone: cross-category links, count
normalization and nonempty conditioning redistribute probability. Inspect predicted
effects instead of equating a coefficient sign with complete behavioral correctness.

### Support and uncertainty

The latest audit used 256 training contexts and retained 53 individual ±10% actions
with enough contexts whose target price was current and whose changed price stayed
inside that product/store's training range. Only one context had all 28 prices
currently observed; about 80.5% of price slots were current. Other prices could be
fallbacks. Staying inside an observed minimum/maximum is a limited support check,
not proof of dense joint support or causal comparability.

Distinguish sampling error, differences between fitted checkpoints, missing/confounded
inputs and an incorrect model structure. Exact integration removes sampling error
from chain expectations. It does not remove the other uncertainties. Two checkpoints'
range is a sensitivity summary, not a calibrated confidence interval.

Promotion scenarios can change price, display and advertising together if those
changes are specified. The CLI currently changes one price at a time; the Python API
can evaluate an explicitly edited context containing multiple changes. The adapter
does not certify price support on its own. Use the matching observed support metadata
before interpreting an application scenario as evidence-backed.

## 12. What each evaluation checks

Every comparison should use the same contexts, target projection, product support,
candidate sets and input information. Models must not obtain true prices or hidden
synthetic states that competitors do not receive. Tuning history and fit budget
must be disclosed. A common finite budget is useful, but does not certify that
every architecture was optimized equally well.

### Whole-basket log likelihood

Average \(\log P(S_i\mid z_i)\) measures how much probability the model gives
the actual entire basket. Higher, usually less negative, is better. This tests joint
prediction, including counts and combinations. A model cannot improve this score
by predicting only the correct mean size. Different supports or conditioning rules
make the score incomparable.

### Item Brier score and calibration

Brier score averages \((\pi_{ij}-y_{ij})^2\) over basket-product pairs. Lower is
better. It checks individual inclusion probabilities. Calibration also groups
predictions into ranges and asks whether a 20% prediction occurs about 20% of the
time. Strong marginal calibration need not imply correct product combinations.

### Recommendation metrics

Pre-basket recall@5 is the fraction of actual selected products appearing among the
top five recommendations, averaged across baskets. NDCG@5 weights a hit higher when
it appears nearer the top, dividing by the ideal score for that basket. The weight
at rank \(r\) is \(1/\log_2(r+1)\). These metrics measure ranking, not probability
calibration or recommendation-induced sales.

In one-item completion, hit@1 and hit@5 record whether the masked product ranks in
those positions. MRR is the mean of \(1/\text{rank of masked product}\). All models
use the same hidden products and eligible candidates; otherwise the comparison
could become easier for one model.

### Generation metrics

For a size histogram, total variation is
\(\tfrac12\sum_n|\widehat P_{\rm generated}(n)-\widehat P_{\rm real}(n)|\).
Zero means identical histogram frequencies. Category-count TV does the same for
the number of present categories. Item-incidence MAE compares product frequencies.
Category-pair MAE compares the probabilities of two categories co-occurring.

Within-category multiple-product rate counts events with at least two selected
products in a category, divided by baskets times categories in this report. It
is not the fraction of whole baskets containing any such event. This is a direct
check that the substitute-group structure is doing useful work.

An observed mean and one generated mean are not enough. Use distribution shapes,
multiple sampling seeds and subgroup checks. Larger catalogues, new stores and new
households require their own evaluations.

### Behavioral and causal checks

In a synthetic world with independent known truth, compare predicted intervention
changes with oracle changes on exactly the same contexts and conditioning. Report
absolute errors, effect magnitudes and calibration slopes, as well as correlations.
A high correlation can coexist with effects twice as large as the truth. A zero
effect can look good when real effects are tiny, so compare with a no-effect baseline.

In real data without intervention truth, a response audit checks direction, support,
numerical accuracy and fit sensitivity. It is not a measurement of causal accuracy.
Household or store-period clustered standard errors reflect dependence within those
groups; they do not correct unobserved confounding.

### Software and optimization checks

Tiny exhaustive examples verify that probabilities sum to one, exact marginals
match summed basket probabilities, and generated baskets follow the same law.
Checkpoint reproduction verifies the new inference code against stored per-basket
likelihoods. Target-blindness checks ensure generated contents do not depend on
the actual target contents or target size.

Projected-gradient convergence is a different check. Passing numerical inference
checks does not imply the fitted checkpoint has converged or behaves correctly.
All these checks answer different questions and should be reported separately.

## 13. What the completed results mean

The latest ERIM development benchmark uses 47,858 training baskets from periods
1–30 and 15,696 validation baskets from periods 31–40. Nearly 99.87% of validation
baskets belong to seen households. This is mainly a known-household prediction
benchmark; it does not establish cold-start performance. Validation has been reused
for development and should not be relabeled an untouched final test.

### Predictive comparison

| Model family | Mean validation basket log likelihood, higher better |
|---|---:|
| Independent products | −3.4545 |
| Historically tuned current chain | −3.6513 |
| Fresh previous coupled-price chain | −3.7384 |
| Fresh revised separated-price chain | −3.7492 |
| Fresh chain without category links | −3.7493 |
| SHOPPER port with shared controls | −3.9560 |
| Popularity | −4.3225 |

The tuned chain had extra historical fitting and is shown separately. Independent
products has about 108,976 parameters, predominantly household-by-product effects;
the current chain has about 16,165 active parameters and four household taste factors.
The stronger baseline suggests that richer preferences and appropriate shrinkage
deserve attention. It does not isolate preference rank as the sole cause, because
capacity, regularization and incomplete optimization all differ.

Removing category links barely changes fresh-chain likelihood. However, the later
generation assessment shows a material category-count distribution difference.
An application-specific benefit should not be discarded just because average
likelihood shows little change.

### Application comparison using saved checkpoints

| Family | Recall@5 ↑ | Completion hit@1 ↑ | Size TV ↓ | Generated mean distinct size |
|---|---:|---:|---:|---:|
| Tuned current chain | 0.6696 | 0.3533 | 0.0111 | 1.3002 |
| Independent products | 0.7076 | 0.3435 | 0.0176 | 1.2989 |
| Fresh revised chain | 0.6534 | 0.3453 | 0.0087 | 1.2928 |
| Previous coupled chain | 0.6542 | 0.3488 | 0.0072 | 1.2842 |
| Chain without category links | 0.6538 | 0.3451 | 0.0221 | 1.2904 |
| SHOPPER port + controls | 0.6510 | 0.3303 | 0.1944 | 1.0440 |

The observed mean is 1.2933. Completion uses 3,721 multi-product baskets. The tuned
chain's small top-one completion advantage over independent products is not a clear
win under the clustered uncertainty checks; independent products wins top-five
completion and marginal recall. SHOPPER has notably weak multi-product likelihood
and produces baskets that are too small in this implementation and fitting setup.
This is evidence about this port, dataset and budget, not a claim that the original
paper can never capture basket size or price effects.

The observed within-category multiple-product rate is 0.002603. Tuned chain gives
0.002492, fresh revised gives 0.002348 and independent products gives 0.007782.
The independent baseline can match mean size while generating about three times
too many same-category multiple-product events. Category structure has useful
generative value even when it loses a recommendation benchmark.

The saved application assessment evaluated 12 checkpoints, generated 376,704 baskets
over two sampling replicas, and reproduced every saved validation likelihood array
exactly using the research adapters. Price scenario evidence remains exploratory:
there is no real counterfactual outcome to score. Tuned-checkpoint relative disagreement
in net-size responses is about 0.59, partly because the mean effects are tiny.
Its absolute mean disagreement is about 0.000895 against mean absolute net response
about 0.001507. Relative instability should be interpreted alongside these magnitudes.

All selected chain fits retain their failed stationarity status. No current result
establishes universally superior prediction, real causal price accuracy, retailer
margin optimization or a complete dynamic twin. Novelty also remains a research
claim to establish, rather than a consequence of combining familiar components.

## 14. From basket simulator to shop twin

The current model is a conditional basket component. A complete shop simulation
would additionally need:

1. Opportunities and visits, including no-purchase and store choice.
2. Units and pack sizes, so item inclusion can become quantity demand.
3. Inventory and availability, with stockouts and substitutions.
4. State across time, such as pantry stock, loyalty and promotion carryover.
5. Reliable intervention evidence and uncertainty for promotion decisions.
6. Costs, funding and constraints for evaluating retailer outcomes.

These components need their own data and validation. Running the static basket model
repeatedly does not create learned dynamics. Updating a shopper's pantry with a
hand-written rule would be an explicit additional assumption.

The approved eventual decision objective is **shop contribution margin**, using
clearly hypothetical product costs until actual costs are available. Under a one-unit
per included product approximation, conditional margin would be

\[
M(z')=\sum_j(p'_j-c_j^{\rm hypothetical})\pi_j(z')-F(z'),
\]

where \(F\) is the stated promotion expense in the same per-basket accounting basis.
Compare a plan's margin with the baseline under the same context population. This
formula is a proposed decision calculation, not a validated margin benchmark. It
does not yield whole-shop profit: selected products, omitted units and missing
visitation effects all limit it. Even a mathematically optimal plan under this
formula can be wrong if price response or costs are wrong.

The next scientific priorities are to improve preference fitting/shrinkage while
preserving both recommendation and generation checks, obtain dependable optimization
diagnostics, and evaluate supported real interventions independently. A fresh final
evaluation must be reserved before further architecture selection. Full dynamics
should follow demonstrated basket and intervention adequacy, with each new component
tested against its own observable outcomes.

## 15. Code, inputs and reproducibility

This standalone release contains the model and training code, not the historical
research directory. It can initialize and train from new data without a checkpoint.
The root README gives the installation and complete data-to-training-to-simulation
commands.

| File | Responsibility |
|---|---|
| `basket_simulator/trainable.py` | Differentiable legacy and anchored models; ESP; full basket likelihood and batched exact sampling |
| `basket_simulator/pipeline.py` | Recommended preference-first, count/category structure-second fitting |
| `basket_simulator/train.py` | Prepared-data validation, bounded full-batch L-BFGS-B, ridge, validation selection, checkpointing and projected-gradient diagnostics |
| `basket_simulator/model.py` | ESP and direct chain sampling; exact small-C or uncertainty-reported large-C predictions, ranking and scenarios |
| `basket_simulator/example_data.py` | Independent procedural software example; no fitted model used to generate targets |
| `basket_simulator/__main__.py` | Application CLI and JSON I/O |
| `tests/` | Exhaustive probability checks, numerical gradients, generation checks and end-to-end fitting |

The standalone trainer preserves the current law and penalized training objective,
with explicit random initialization and unscaled L-BFGS-B coordinates. It does not
replay every historical continuation, block-scaling strategy or fit budget. Numerical
law equivalence is distinct from replicating historical optimizer trajectories.
The training input is a prepared NPZ with context arrays plus binary `member` targets.
All models use a uniform category width and assume full catalogue availability.
The original raw retail preprocessing and ID mapping must be reproduced by the
researcher for their own dataset. This package does not infer a reliable price feed
from purchase-only transactions automatically.

Training writes `protocol.json`, `input_manifest.json`, `history.json`, `fit.json`,
`config.json`, `best.pt`, `latest.pt` and `final.pt`. Best parameters are chosen by
validation likelihood; best and final stationarity are checked separately. The
trainer does not automatically declare a budget-stopped model converged. Preserve
an untouched final split and evaluate after architecture/model selection is complete.

A context JSON contains household/store internal IDs, numeric week and C-by-P arrays
`dlp`, `disp`, `mail`. Price features use the fixed training reference. No target
basket content or size is supplied for inference. Predictions return category/item
probabilities and expected distinct size; generation returns binary basket arrays;
scenarios return before/after differences. Row0 is reserved for unknown IDs.

The example data generator supplies random observed covariates, independent category
needs conditioned nonempty, mostly one-product counts and price-sensitive product
choices without replacement. It is a runnable software demonstration, not a
comprehensive robustness test or real intervention benchmark. The inference-only
`demo` command is a separate hand-set illustration, not a trained fit.

The measured results in sections13 and11 are inherited from completed ERIM research
runs, not from training the procedural example. Their source artifacts were
`research_erim_baselines_20261012/result.json` and
`research_checkpoint_applications_20261012/{application_comparison.csv,result.json,
interpretation.json,plan_manifest.json}`. Those local data/checkpoints are not
shipped with this clean model release. No training-convergence or causal claim is
improved simply by repackaging the implementation.

The older Version-4 global energy model and its Gaussian/Smolyak pipeline are not
part of this release. The finite category-chain/ESP law above is the model being
trained here. A new reader only needs the README, this guide and the files in the
folder map; historical handoff notes are deliberately excluded.
