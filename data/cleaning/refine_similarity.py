from pathlib import Path
from PIL import Image,ImageOps
import csv,json,math,random,shutil
import numpy as np
from collections import defaultdict,Counter
p=Path(__import__('sys').argv[1]).resolve()
assert p.name=='scoliosis_depurado' and p.parent.name=='outputs'
rows=list(csv.DictReader((p/'manifest.csv').open(encoding='utf-8-sig'))); active=[r for r in rows if r['status']=='retained']; parent=list(range(len(active)))
def find(i):
 while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
 return i
def union(a,b): parent[find(a)]=find(b)
seen={}; hashes=[]
for i,r in enumerate(active):
 if r['group'] in seen: union(i,seen[r['group']])
 else: seen[r['group']]=i
 with Image.open(p/r['split']/'images'/r['name']) as im:
  im.draft('L',(64,64)); a=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS));
  def dh(a):
   v=0
   for bit in (a[:,1:]>a[:,:-1]).ravel(): v=(v<<1)|int(bit)
   return v
  hashes.append((dh(a),dh(a[:,::-1]),int(r['width'])/int(r['height'])))
print('Perceptual hashes calculated',flush=True)
pairs=[]
for i,(h,m,ratio) in enumerate(hashes):
 for j in range(i):
  hh,mm,rr=hashes[j]
  if abs(math.log(ratio/rr))>.05: continue
  distance=min((h^hh).bit_count(),(h^mm).bit_count())
  if distance<=4:
   if find(i)!=find(j): pairs.append(dict(image_a=active[i]['name'],image_b=active[j]['name'],distance=distance,old_split_a=active[i]['split'],old_split_b=active[j]['split']))
   union(i,j)
print('Additional similarity links',len(pairs),flush=True)
groups=defaultdict(list)
for i,r in enumerate(active): r['group']=f'p{find(i):05d}'; groups[r['group']].append(i)
vectors={}; totals=np.zeros(5)
for g,ids in groups.items():
 v=np.zeros(5); v[4]=len(ids)
 for i in ids:
  r=active[i]; lines=(p/r['split']/'labels'/Path(r['name']).with_suffix('.txt')).read_text().splitlines()
  r['_boxes']=[list(map(float,line.split())) for line in lines]
  for c in {int(b[0]) for b in r['_boxes']}: v[c]+=1
 vectors[g]=v; totals+=v
splits=['train','test','val']; target=np.array([.7,.25,.05])[:,None]*totals; best=None
for trial in range(120):
 rng=random.Random(20261114+trial); order=list(groups); rng.shuffle(order); order.sort(key=lambda g:len(groups[g]),reverse=True); counts=np.zeros((3,5)); assign={}
 for g in order:
  scores=[]
  for s in range(3):
   candidate=counts.copy(); candidate[s]+=vectors[g]; scores.append(float(np.sum((candidate-target)**2/np.maximum(target,1))))
  s=min(range(3),key=lambda s:scores[s]); counts[s]+=vectors[g]; assign[g]=splits[s]
 score=float(np.sum((counts-target)**2/np.maximum(target,1)))
 if best is None or score<best[0]: best=(score,assign)
assign=best[1]
for r in active:
 old=r['split']; new=assign[r['group']]
 if new!=old:
  for sub,filename in [('images',r['name']),('labels',Path(r['name']).with_suffix('.txt'))]:
   source=(p/old/sub/filename).resolve(); dest=(p/new/sub/filename).resolve(); assert p in source.parents and p in dest.parents and not dest.exists(); shutil.move(str(source),str(dest))
 r['split']=new
for key in ['sha256','pixel_hash','mirror_key','source_key','group']:
 seen=defaultdict(set)
 for r in active: seen[r[key]].add(r['split'])
 assert all(len(v)==1 for v in seen.values())
for pair in pairs:
 a=next(r for r in active if r['name']==pair['image_a']); b=next(r for r in active if r['name']==pair['image_b']); assert a['split']==b['split']
fields=[k for k in rows[0] if not k.startswith('_')]
with (p/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
(p/'similarity_links.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
audit=json.loads((p/'audit.json').read_text()); audit['known_source_groups']=len(groups); audit['perceptual_similarity_links']=len(pairs); audit['perceptual_method']='64-bit dHash on 9x8 grayscale, including horizontal reflection; Hamming distance <=4 and absolute log-aspect-ratio difference <=0.05. Conservative group union; no images deleted for perceptual similarity.'
audit['summary']={s:dict(images=sum(r['split']==s for r in active),groups=sum(assign[g]==s for g in groups),boxes_by_class=dict(Counter(int(b[0]) for r in active if r['split']==s for b in r['_boxes']))) for s in splits}
audit['limitations'][2]='Perceptual similarity grouping is heuristic: may merge distinct images or miss related images; it is not a verified patient identifier.'
(p/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
readme=(p/'README.md').read_text(); readme=readme[:readme.index('{\n')]; readme+='## Segunda comprobacion de similitud\nSe aplica dHash de 64 bits con reflejo horizontal, distancia Hamming <=4 y diferencia logaritmica de relacion de aspecto <=0.05. Se unen grupos transitivamente sin eliminar imagenes por similitud. Esto puede agrupar imagenes diferentes y omitir variantes; no equivale a identificacion de pacientes. Las relaciones quedan en similarity_links.json.\nPara reproducir TODO: ejecutar prepare_dataset.py primero y luego python refine_similarity.py CARPETA_DATASET. Ejecutar el segundo paso solo una vez.\n\n'+json.dumps(audit,indent=2)
(p/'README.md').write_text(readme,encoding='utf-8'); shutil.copy2(__file__,p/'refine_similarity.py')
# Regenerate the preview using final assignments.
from PIL import ImageDraw
sheet=Image.new('RGB',(1200,680),'white'); d=ImageDraw.Draw(sheet); chosen=[]
for c in range(4): chosen.extend([r for r in active if any(int(b[0])==c for b in r['_boxes'])][:2])
for k,r in enumerate(chosen):
 im=Image.open(p/r['split']/'images'/r['name']).convert('RGB'); im.thumbnail((290,290)); x=(k%4)*300+(300-im.width)//2; y=(k//4)*340+25; sheet.paste(im,(x,y))
 for c,cx,cy,w,h in r['_boxes']: d.rectangle((x+(cx-w/2)*im.width,y+(cy-h/2)*im.height,x+(cx+w/2)*im.width,y+(cy+h/2)*im.height),outline=['red','lime','cyan','orange'][int(c)],width=2)
 d.text(((k%4)*300+8,(k//4)*340+5),f"class {int(r['_boxes'][0][0])} | {r['split']}",fill='black')
sheet.save(p/'annotation_preview.jpg')
print(json.dumps(audit,indent=2),flush=True)
shutil.make_archive(str(p.parent/'scoliosis_depurado'),'zip',p.parent,p.name)
print('Updated ZIP complete',flush=True)
