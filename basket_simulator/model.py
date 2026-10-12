"""Exact, single-context reference inference; no research-repository imports.

Enumeration is deliberate: this application adapter targets the saved 7x4 model,
not full-store fitting. The research implementation retains ESP dynamic programs.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def patterns(n):
    return np.array(list(itertools.product((0., 1.), repeat=n)))


class BasketSimulator:
    def __init__(self, state, config):
        self.s = {k: np.asarray(v) for k, v in state.items()}
        self.config = config
        self.C, self.P, self.R = (int(config[k]) for k in ("C", "P", "R"))
        if not (1 <= self.C <= 10 and 1 <= self.R <= self.P <= 10):
            raise ValueError("Reference inference supports C,P <=10 and 1<=R<=P")
        if any(not np.isfinite(v).all() for v in self.s.values()):
            raise ValueError("Checkpoint parameters must be finite")
        if not np.array_equal(np.sort(self.s["order"]),np.arange(self.C)):
            raise ValueError("Category order must be a permutation")
        support = self.s["calendar_support"]
        if support.shape != (2,) or support[0] > support[1]:
            raise ValueError("Calendar support needs two ordered endpoints")
        # Reject latent/nested/history variants that this adapter does not implement.
        fixed = {"gamma": 0., "history_coefficient": 0., "mu_single": 1.,
                 "mu_multi": 1., "log_taste_scale": 0., "log_price_single": 0.,
                 "log_price_multi": 0., "history_single": 0., "history_multi": 0.,
                 "item_price_loading": 1., "price_sd": 0.,
                 "count_price_gate": 1., "category_price_gate": 1.}
        for k, value in fixed.items():
            if k in self.s and not np.allclose(self.s[k], value, atol=1e-12, rtol=0):
                raise ValueError(f"Unsupported active extension: {k}")
        if not np.array_equal(self.s["layout"], np.arange(self.C*self.P).reshape(self.C,self.P)):
            raise ValueError("Expected the current contiguous, fully offered catalogue")
        if "choice_mapping" in self.s and np.any(self.s["choice_mapping"] != 0):
            raise ValueError("Nested choice groups are unsupported")
        for k in ("choice_price", "category_price", "count_price_scale"):
            if not np.isfinite(self.s[k]).all() or (self.s[k] < 0).any():
                raise ValueError(f"Invalid nonnegative parameter: {k}")
        self.cats = patterns(self.C)
        subsets = patterns(self.P)[1:]
        self.subsets = subsets[subsets.sum(1) <= self.R]
        self.sizes = self.subsets.sum(1).astype(int)

    @classmethod
    def load(cls, checkpoint, config):
        import torch  # Needed only for loading existing research checkpoints.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return cls({k: v.detach().numpy() for k,v in state.items()},
                   json.loads(Path(config).read_text()))

    def tables(self, context):
        required = {"house", "store", "week", "dlp", "disp", "mail"}
        if set(context) != required:
            raise ValueError(f"Context must contain exactly {sorted(required)}; no basket targets")
        s = self.s
        h, store = context["house"], context["store"]
        if not isinstance(h, (int,np.integer)) or not 0 <= h < len(s["theta"]):
            raise ValueError("house must be a valid internal integer ID")
        if not isinstance(store, (int,np.integer)) or not 0 <= store < len(s["xi"]):
            raise ValueError("store must be a valid internal integer ID")
        fields = [np.asarray(context[k], dtype=float) for k in ("dlp","disp","mail")]
        if any(v.shape != (self.C,self.P) or not np.isfinite(v).all() for v in fields):
            raise ValueError("dlp/disp/mail require finite C-by-P arrays")
        if not np.isfinite(context["week"]):
            raise ValueError("week must be finite")
        price, display, mail = fields
        phase = (np.clip(context["week"], *s["calendar_support"])-1)*2*np.pi/52
        calendar = np.array([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)])
        b = (s["lam"]+s["theta"][h]@s["alpha"].T+s["xi"][store]@s["zeta"].T
             +calendar@s["mu"].T).reshape(self.C,self.P)
        b += s["w_dsp"].reshape(self.C,self.P)*display+s["w_mlr"].reshape(self.C,self.P)*mail
        u = b-s["choice_price"].reshape(self.C,self.P)*price
        ref = b@self.subsets.T
        actual = u@self.subsets.T
        log_ref = np.stack([logsumexp(ref[:,self.sizes==n],axis=1) for n in range(1,self.R+1)],1)
        log_esp = np.stack([logsumexp(actual[:,self.sizes==n],axis=1) for n in range(1,self.R+1)],1)
        n = np.arange(1,self.R+1)
        penalty = s["rho"][:,None]*(n*(n-1)/2)
        a0 = log_ref-penalty
        a = a0+s["count_price_scale"][:,None]*(log_esp-log_ref)
        log_count = a-logsumexp(a,axis=1,keepdims=True)
        log_within = log_count[:,self.sizes-1]+actual-log_esp[:,self.sizes-1]
        iv = (logsumexp(a0,axis=1)-(s["category_price"]*price).sum(1)
              if bool(s["separate_category_price"]) else logsumexp(log_esp-penalty,axis=1))
        cross = s["cross_p"]@s["cross_q"].T
        np.fill_diagonal(cross,0.)
        base = s["kappa"]+np.logaddexp(0.,s["tau"])*iv+cross@iv+s["U"]@s["theta"][h]
        order = s["order"].astype(int)
        x = self.cats[:,order]
        contrib = x[...,None]*s["W"]
        hidden = np.maximum(0.,np.cumsum(contrib,axis=1)-contrib+s["h0"])
        count = np.cumsum(x,axis=1)-x
        eta = (base[order]+x@np.tril(s["V"],-1).T+(hidden*s["omega"]).sum(-1)
               +s["psi0"][count.astype(int)]+s["psi1"]*count)
        log_mass = (-x*np.logaddexp(0.,-eta)-(1-x)*np.logaddexp(0.,eta)).sum(1)
        log_mass[0] = -np.inf
        log_mass -= logsumexp(log_mass)
        return log_mass, log_within

    def predict(self, context):
        mass, within = (np.exp(t) for t in self.tables(context))
        category = mass@self.cats
        incidence = category[:,None]*(within@self.subsets)
        return {"incidence": incidence, "category_presence": category,
                "expected_distinct_size": float(incidence.sum())}

    def log_probability(self, context, basket):
        y = np.asarray(basket)
        if y.shape != (self.C,self.P) or not np.isin(y,[0,1]).all() or not y.any():
            raise ValueError("basket must be a nonempty binary C-by-P array")
        cats = y.any(1)
        if (y.sum(1)>self.R).any():
            return -np.inf
        mass, within = self.tables(context)
        idx = np.flatnonzero((self.cats==cats).all(1))[0]
        result = mass[idx]
        for c in np.flatnonzero(cats):
            i = np.flatnonzero((self.subsets==y[c]).all(1))[0]
            result += within[c,i]
        return float(result)

    def generate(self, context, draws=1, seed=0):
        if not isinstance(draws,int) or draws<1:
            raise ValueError("draws must be a positive integer")
        mass, within = (np.exp(t) for t in self.tables(context))
        rng = np.random.default_rng(seed)
        cats = self.cats[rng.choice(len(mass),draws,p=mass)]
        out = np.zeros((draws,self.C,self.P),dtype=bool)
        for c in range(self.C):
            out[:,c] = self.subsets[rng.choice(len(self.subsets),draws,p=within[c])]*cats[:,c,None]
        return out

    def recommend(self, context, revealed=None):
        if revealed is None:
            scores = self.predict(context)["incidence"].ravel()
            return np.argsort(-scores,kind="stable"), scores
        y = np.asarray(revealed).copy()
        if y.shape != (self.C,self.P) or not np.isin(y,[0,1]).all():
            raise ValueError("revealed must be a binary C-by-P array")
        scores = np.full(self.C*self.P,-np.inf)
        for j in np.flatnonzero(~y.astype(bool).ravel()):
            candidate = y.copy(); candidate.flat[j] = 1
            scores[j] = self.log_probability(context,candidate)
        candidates = np.flatnonzero(np.isfinite(scores))
        return candidates[np.argsort(-scores[candidates],kind="stable")], scores

    def what_if(self, context, product, factor):
        if not isinstance(product,int) or not 0<=product<self.C*self.P:
            raise ValueError("product must be an internal integer product ID")
        if not np.isfinite(factor) or factor<=0:
            raise ValueError("price factor must be positive and finite")
        changed = {**context,"dlp":np.array(context["dlp"],dtype=float,copy=True)}
        changed["dlp"].flat[product] += np.log(factor)
        before, after = self.predict(context), self.predict(changed)
        return {"before":before,"after":after,
                "incidence_change":after["incidence"]-before["incidence"],
                "size_change":after["expected_distinct_size"]-before["expected_distinct_size"]}


def toy_model():
    """Illustrative parameters, NOT a fitted model or a scientific benchmark."""
    C,P,J = 2,3,6
    s = {"lam":np.array([.4,.1,-.2,.2,0.,-.3]),"theta":np.zeros((1,1)),
         "alpha":np.zeros((J,1)),"xi":np.zeros((1,1)),"zeta":np.zeros((J,1)),
         "mu":np.zeros((J,4)),"w_dsp":np.zeros(J),"w_mlr":np.zeros(J),
         "rho":np.full(C,2.),"choice_price":np.ones(J),"category_price":np.full((C,P),.2),
         "count_price_scale":np.full(C,.2),"calendar_support":np.array([1,30]),
         "separate_category_price":np.array(True),"layout":np.arange(J).reshape(C,P),
         "kappa":np.full(C,-1.),"tau":np.zeros(C),"cross_p":np.zeros((C,1)),
         "cross_q":np.zeros((C,1)),"U":np.zeros((C,1)),"order":np.arange(C),
         "V":np.array([[0.,0.],[.4,0.]]),"W":np.zeros((C,1)),"omega":np.zeros((C,1)),
         "h0":np.zeros(1),"psi0":np.zeros(C+1),"psi1":np.zeros(C)}
    context = {"house":0,"store":0,"week":10,
               **{k:np.zeros((C,P)).tolist() for k in ("dlp","disp","mail")}}
    return BasketSimulator(s,{"C":C,"P":P,"R":P}),context
