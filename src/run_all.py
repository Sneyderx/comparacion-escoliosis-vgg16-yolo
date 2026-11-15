"""Fresh Python process per model/seed. Train all models before testing any."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['train','evaluate','smoke'],required=True); p.add_argument('--dataset',required=True); p.add_argument('--results',default='results'); a=p.parse_args(); cfg=json.loads(Path('config.json').read_text(encoding='utf-8-sig'))
if a.stage=='evaluate':
 missing=[str(Path(a.results)/'experiment'/m/f'seed_{s}'/'training_done.json') for m in cfg['models'] for s in cfg['seeds'] if not (Path(a.results)/'experiment'/m/f'seed_{s}'/'training_done.json').exists()]
 if missing: raise RuntimeError('Finish all training before testing: '+str(missing))
for seed in (cfg['seeds'][:1] if a.stage=='smoke' else cfg['seeds']):
 # Alternate order to reduce a constant ordering bias in runtime measurements.
 for model in (cfg['models'] if cfg['seeds'].index(seed)%2==0 else list(reversed(cfg['models']))):
  cmd=[sys.executable,'src/run.py','--mode','train' if a.stage=='smoke' else a.stage,'--model',model,'--seed',str(seed),'--dataset',a.dataset,'--results',a.results]
  if a.stage=='smoke': cmd.append('--smoke')
  subprocess.run(cmd,check=True)
