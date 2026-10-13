"""Fresh-process actual fitted-checkpoint timing and RSS."""
import json,time,sys,resource,statistics,platform
from . import common as c
from . import evaluate as e

def main(name):
 from .runtime import install
 install()
 m=c.load_model(name);data=c.Data('test');a=e.Adapter(m);ctx,y=data.batch(slice(0,128));cache=a.cache(ctx)
 def timed(fn):
  fn();rows=[]
  for _ in range(5):
   t=time.monotonic();fn();rows.append(time.monotonic()-t)
  return {'median_seconds':statistics.median(rows),'repeats':rows}
 import torch
 with torch.no_grad():ll=timed(lambda:m.log_probability(ctx,y))
 x={k:v[:16].repeat_interleave(16,0) for k,v in cache.items()};generation=timed(lambda:a.sample(x,972901))
 before,_=data.batch(slice(0,16));after={**before,'dlp':before['dlp'].clone()};after['dlp'][:,0,0]+=e.np.log(.9)
 def scenario():
  left=a.cache(before);right=a.cache(after);a.sample({k:v.repeat_interleave(128,0) for k,v in left.items()},972902);a.sample({k:v.repeat_interleave(128,0) for k,v in right.items()},972902)
 whatif=timed(scenario)
 row={'model':name,'total_stored_parameters':sum(p.numel() for p in m.parameters()),'active_last_stage_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if platform.system()=='Darwin' else 1024),'likelihood128':ll,'generate256':generation,'paired_scenario16contexts128draws':whatif,'scope':'Fresh process including common imports/data/checkpoint and application workspace; excludes optimizer history. Training timings in fits_complete and fit_records.'}
 c.write(name+'_practical.json',row);print(json.dumps(row))
if __name__=='__main__':main(sys.argv[1])
