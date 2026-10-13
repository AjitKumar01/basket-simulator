"""One readable report for all five evaluation panels."""

import csv
import html
import json
from . import common as c


def render(result):
    audit = json.loads((c.OUT / "data_audit.json").read_text())
    cfg = json.loads((c.OUT / "data/config.json").read_text())
    fits = json.loads((c.OUT / "fit_records.json").read_text())
    metrics = [
        "basket_ll",
        "recall5",
        "recall10",
        "size_tv",
        "product_frequency_mae",
        "category_pair_mae",
        "brier",
        "generated_mean_size",
    ]
    records = [{"model": name, **row} for name, row in result["models"].items()]
    with (c.OUT / "real_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    cells = "".join("<th>" + html.escape(k) + "</th>" for k in ["model"] + metrics)
    rows = "".join(
        "<tr><td>"
        + html.escape(name)
        + "</td>"
        + "".join(f"<td>{row[k]:.6f}</td>" for k in metrics)
        + "</tr>"
        for name, row in result["models"].items()
    )
    gates = "".join(
        f'<tr><td>{html.escape(name)}</td><td>{row.get("final_stationarity_passed")}</td><td>{row.get("validation_ll")}</td></tr>'
        for name, row in fits.items()
    )
    text = f"""<!doctype html><meta charset="utf-8"><title>Basket comparison</title>
    <style>body{{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}</style>
    <h1>Basket comparison: {cfg['C']*cfg['P']} products, {cfg['C']} categories</h1>
    <p>Training: {audit['splits']['train']['baskets_projected']} baskets. Likelihood scores every later basket; generation and recommendation share contexts and draws. Conditional nonempty distinct products.</p>
    <h2>1. Predictive fit and 2. Basket generation</h2><table><tr>{cells}</tr>{rows}</table>
    <p><a href="comparison.json">Metrics, household uncertainty and Monte Carlo replicas</a> · <a href="store_period_sensitivity.json">Separate store-period sensitivity</a> · <a href="real_metrics.csv">CSV</a></p>
    <h2>3. Price scenarios</h2><p><a href="scenarios.json">Identical discounts: own, sibling, cross-category, category participation and basket-size predictions</a>. Real causal accuracy is unavailable; observational price proxies do not supply intervention truth.</p>
    <h2>4. Promotion decisions</h2><p>The scenario file records global plan selection under six explicitly hypothetical cost/funding settings. Actual real margin and regret are unavailable.</p>
    <h2>5. Practical scalability</h2><p><a href="practical.json">Five-repeat fresh-process application timings and RSS</a> · <a href="fits_complete.json">Training time and peak RSS</a>. Optimizer history is excluded from application RSS.</p>
    <h2>Convergence and scope</h2><table><tr><th>Model</th><th>Final stage passed</th><th>Validation likelihood</th></tr>{gates}</table>
    <p><a href="fit_records.json">Fit diagnostics</a> · <a href="protocol.json">Protocol</a> · <a href="data_audit.json">Data audit</a>. Final convergence and selected-checkpoint convergence are separate; no global optimum or causal certificate. Unsupported optional baselines remain flagged.</p>"""
    if (c.OUT / "oracle_comparison.json").exists():
        text += '<h2>Synthetic known-truth evaluation</h2><p><a href="oracle_comparison.json">Exact generator generation, price-response accuracy and hypothetical promotion regret</a>. Generator truth is never used in training or selection.</p>'
    (c.OUT / "comparisons.html").write_text(text)
