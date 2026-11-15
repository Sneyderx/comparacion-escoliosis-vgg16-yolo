"""Validate single-class image labels and preserve the cleaned grouped split."""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

def prepare(root, output):
 root=Path(root).resolve(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
 source=list(csv.DictReader((root/'manifest.csv').open(encoding='utf-8-sig')))
 rows=[]; groups=defaultdict(set); seen=defaultdict(set)
 for r in source:
  if r['status']!='retained': continue
  split=r['split']; image=root/split/'images'/r['name']; label=root/split/'labels'/Path(r['name']).with_suffix('.txt')
  assert image.is_file() and label.is_file(), image
  classes={int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
  if len(classes)!=1: raise ValueError(f'Image without a unique class: {image}: {classes}')
  cls=next(iter(classes)); assert 0<=cls<4
  sha=hashlib.sha256(image.read_bytes()).hexdigest(); assert sha==r['sha256'], image
  groups[r['group']].add(split); seen[sha].add(split)
  rows.append(dict(relative_path=image.relative_to(root).as_posix(),split=split,class_id=cls,group=r['group'],sha256=sha))
 assert all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in seen.values())
 assert {r['split'] for r in rows}=={'train','val','test'}
 for s in ['train','val','test']: assert {r['class_id'] for r in rows if r['split']==s}==set(range(4))
 rows.sort(key=lambda r:r['relative_path'])
 path=output/'classification.csv'
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,rows[0].keys()); w.writeheader(); w.writerows(rows)
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 info=dict(images=len(rows),by_split=dict(Counter(r['split'] for r in rows)),by_split_and_class={s:dict(Counter(r['class_id'] for r in rows if r['split']==s)) for s in ['train','val','test']},manifest_sha256=digest,class_names=['1-derece','2-derece','3-derece','saglikli'],task='image classification; full radiographs; no bounding-box input at inference')
 (output/'classification_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8'); print(json.dumps(info,indent=2)); return info
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--output',default='data'); a=p.parse_args(); prepare(a.dataset,a.output)
