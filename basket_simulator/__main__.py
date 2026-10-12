"""Small application CLI; run python -m basket_simulator --help."""
import argparse
import json
from pathlib import Path
import numpy as np
from .model import BasketSimulator, toy_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["demo","predict","generate","recommend","what-if"])
    p.add_argument("--checkpoint");p.add_argument("--config");p.add_argument("--context")
    p.add_argument("--revealed",help="JSON binary C-by-P array; assumes exactly one missing item")
    p.add_argument("--draws",type=int,default=5);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--mc-draws",type=int,default=8192,help="Category integration draws when C>10")
    p.add_argument("--product",type=int,default=0);p.add_argument("--factor",type=float,default=.9)
    p.add_argument("--output",help="Write JSON to this file instead of stdout")
    args = p.parse_args()
    if args.action=="demo":
        m,ctx = toy_model()
        result = {"status":"illustrative hand-set parameters; not trained",
                  "prediction":m.predict(ctx),"baskets":m.generate(ctx,args.draws,args.seed),
                  "discount":m.what_if(ctx,args.product,args.factor)}
    else:
        if not all((args.checkpoint,args.config,args.context)):
            p.error("--checkpoint, --config and --context are required")
        m = BasketSimulator.load(args.checkpoint,args.config)
        ctx = json.loads(Path(args.context).read_text())
        if args.action=="predict":result=m.predict(ctx,draws=args.mc_draws,seed=args.seed)
        elif args.action=="generate":result=m.generate(ctx,args.draws,args.seed)
        elif args.action=="what-if":result=m.what_if(ctx,args.product,args.factor,draws=args.mc_draws,seed=args.seed)
        else:
            revealed = json.loads(Path(args.revealed).read_text()) if args.revealed else None
            order,scores = m.recommend(ctx,revealed,draws=args.mc_draws,seed=args.seed)
            result={"ranked_internal_product_ids":order,
                    "scores":[float(s) if np.isfinite(s) else None for s in scores],
                    "score_type":"joint log probability" if revealed is not None else "inclusion probability"}
    output = json.dumps(result,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a,
                        indent=2,allow_nan=False)+"\n"
    if args.output:Path(args.output).write_text(output)
    else:print(output,end="")


if __name__=="__main__":main()
