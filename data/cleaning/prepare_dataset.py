import zipfile, hashlib, json, csv, random, math, argparse, shutil
from pathlib import Path
from collections import Counter,defaultdict
from io import BytesIO
import numpy as np
from PIL import Image,ImageOps,ImageDraw

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
 out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.csv').exists(): raise RuntimeError('Output already populated; use a fresh output directory')
 z=zipfile.ZipFile(args.input); records=[]; changes=[]
 for n in sorted(z.namelist()):
  if '/images/' not in n or not n.lower().endswith('.jpg'): continue
  lab=n.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'; raw=z.read(n); boxes=[]; conversions=0
  for line in z.read(lab).decode().splitlines():
   a=line.split(); c=int(a[0]); v=list(map(float,a[1:])); assert c in range(4) and all(math.isfinite(x) and 0<=x<=1 for x in v)
   if len(v)>4 and len(v)%2==0:
    xs=v[::2]; ys=v[1::2]; v=[(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,max(xs)-min(xs),max(ys)-min(ys)]; conversions+=1
   assert len(v)==4 and v[2]>0 and v[3]>0
   x1,y1,x2,y2=v[0]-v[2]/2,v[1]-v[3]/2,v[0]+v[2]/2,v[1]+v[3]/2
   if x1<0 or y1<0 or x2>1 or y2>1:
    changes.append(dict(source=n,action='clip_box_to_image',before=v.copy()))
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(1,x2),min(1,y2)
    v=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
    assert v[2]>0 and v[3]>0
   boxes.append((c,*v))
  with Image.open(BytesIO(raw)) as im:
   im.load(); rgb=im.convert('RGB'); pix=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(); flip=hashlib.sha256(str(rgb.size).encode()+ImageOps.mirror(rgb).tobytes()).hexdigest(); size=rgb.size
  r=dict(source=n,old_split=n.split('/')[-3],name=Path(n).name,source_key=Path(n).name.split('.rf.')[0],sha256=hashlib.sha256(raw).hexdigest(),pixel_hash=pix,mirror_key=min(pix,flip),width=size[0],height=size[1],boxes=sorted(set(boxes)),conversions=conversions)
  records.append(r)
  if conversions: changes.append(dict(source=n,action='polygon_to_enclosing_bbox',count=conversions))
 print('Read',len(records),'images',flush=True)
 parent=list(range(len(records)))
 def find(i):
  while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
  return i
 def union(a,b): parent[find(a)]=find(b)
 for key in ['source_key','sha256','pixel_hash','mirror_key']:
  seen={}
  for i,r in enumerate(records):
   if r[key] in seen: union(i,seen[r[key]])
   else: seen[r[key]]=i
 # Quarantine identical decoded images with incompatible annotations; no invented adjudication.
 exact=defaultdict(list)
 for i,r in enumerate(records): exact[r['pixel_hash']].append(i)
 excluded=set()
 for ids in exact.values():
  ref=records[ids[0]]['boxes']
  conflict=any(len(records[i]['boxes'])!=len(ref) or any(a[0]!=b[0] or max(abs(x-y) for x,y in zip(a[1:],b[1:]))>0.01 for a,b in zip(ref,records[i]['boxes'])) for i in ids)
  if conflict:
   for i in ids: excluded.add(i); changes.append(dict(source=records[i]['source'],action='quarantine_conflicting_annotations'))
 # Keep one file per identical decoded pixel array, while preserving known source groups.
 retained=[]; seen={}
 for i,r in enumerate(records):
  r['group']=f'g{find(i):05d}'
  if i in excluded: r['status']='quarantine'; continue
  if r['pixel_hash'] in seen:
   r['status']='duplicate'; r['duplicate_of']=records[seen[r['pixel_hash']]]['source']; continue
  seen[r['pixel_hash']]=i; r['status']='retained'; retained.append(i)
 groups=defaultdict(list)
 for i in retained: groups[records[i]['group']].append(i)
 splits=['train','test','val']; ratios=np.array([.70,.25,.05]); totals=np.zeros(5)
 vectors={}
 for g,ids in groups.items():
  v=np.zeros(5); v[4]=len(ids)
  for i in ids:
   for c in {b[0] for b in records[i]['boxes']}: v[c]+=1
  vectors[g]=v; totals+=v
 target=ratios[:,None]*totals[None,:]; best=None
 for trial in range(120):
  rng=random.Random(20261114+trial); order=list(groups); rng.shuffle(order); order.sort(key=lambda g:len(groups[g]),reverse=True)
  counts=np.zeros((3,5)); assign={}
  for g in order:
   scores=[]
   for s in range(3):
    candidate=counts.copy(); candidate[s]+=vectors[g]
    scores.append(float(np.sum((candidate-target)**2/np.maximum(target,1))))
   s=min(range(3),key=lambda s:scores[s]); counts[s]+=vectors[g]; assign[g]=splits[s]
  score=float(np.sum((counts-target)**2/np.maximum(target,1)))
  if best is None or score<best[0]: best=(score,assign,counts)
 assignment=best[1]
 for i in retained: records[i]['split']=assignment[records[i]['group']]
 for split in splits:
  (out/split/'images').mkdir(parents=True,exist_ok=True); (out/split/'labels').mkdir(parents=True,exist_ok=True)
 for i in retained:
  r=records[i]; (out/r['split']/'images'/r['name']).write_bytes(z.read(r['source']))
  text='\n'.join(' '.join([str(b[0])]+[f'{v:.10f}' for v in b[1:]]) for b in r['boxes'])+'\n'
  (out/r['split']/'labels'/Path(r['name']).with_suffix('.txt')).write_text(text,encoding='utf-8')
 (out/'data.yaml').write_text("# Paths relative to this YAML; dataset root must be the YAML directory.\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 4\nnames: ['1-derece', '2-derece', '3-derece', 'saglikli']\n",encoding='utf-8')
 fields=['source','old_split','name','source_key','sha256','pixel_hash','mirror_key','group','status','split','duplicate_of','width','height','conversions']
 with (out/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(records)
 (out/'changes.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
 summary={s:dict(images=sum(records[i]['split']==s for i in retained),groups=sum(assignment[g]==s for g in groups),boxes_by_class=dict(Counter(b[0] for i in retained if records[i]['split']==s for b in records[i]['boxes']))) for s in splits}
 # Verify written contents and all grouping guarantees.
 for key in ['sha256','pixel_hash','mirror_key','source_key','group']:
  check=defaultdict(set)
  for i in retained: check[records[i][key]].add(records[i]['split'])
  assert all(len(v)==1 for v in check.values()), key
 for i in retained:
  r=records[i]; p=out/r['split']/'images'/r['name']; assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
  for line in (out/r['split']/'labels'/Path(r['name']).with_suffix('.txt')).read_text().splitlines(): assert len(line.split())==5
 report=dict(input_images=len(records),retained_images=len(retained),removed_duplicates=sum(r['status']=='duplicate' for r in records),quarantined_images=len(excluded),polygon_annotations_converted=sum(r['conversions'] for r in records),boxes_clipped=sum(c['action']=='clip_box_to_image' for c in changes),known_source_groups=len(groups),seed=20261114,target_ratios=dict(zip(splits,ratios)),summary=summary,checks='Passed: image integrity, normalized boxes, matching labels; no overlap of file/pixel/mirrored-pixel hashes, source keys, or groups between splits.',limitations=['No verified patient identifiers: patient independence cannot be guaranteed.','Existing augmentation cannot reliably be reversed. Validation/test may include augmented images.','Near duplicates beyond exact pixels, exact horizontal mirrors and source names are not exhaustively detected.','Clinical class definitions and annotation accuracy require expert/source validation.'])
 (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 # Contact sheet: two examples per class; labels retain original names.
 chosen=[]
 for c in range(4): chosen.extend([i for i in retained if any(b[0]==c for b in records[i]['boxes'])][:2])
 sheet=Image.new('RGB',(1200,680),'white'); d=ImageDraw.Draw(sheet)
 for k,i in enumerate(chosen):
  r=records[i]; im=Image.open(out/r['split']/'images'/r['name']).convert('RGB'); im.thumbnail((290,290)); x=(k%4)*300+(300-im.width)//2; y=(k//4)*340+25; sheet.paste(im,(x,y))
  for b in r['boxes']:
   c,cx,cy,w,h=b; d.rectangle((x+(cx-w/2)*im.width,y+(cy-h/2)*im.height,x+(cx+w/2)*im.width,y+(cy+h/2)*im.height),outline=['red','lime','cyan','orange'][c],width=2)
  d.text(((k%4)*300+8,(k//4)*340+5),f"class {r['boxes'][0][0]} | {r['split']}",fill='black')
 sheet.save(out/'annotation_preview.jpg')
 readme='''# Dataset depurado de escoliosis\n\nFuente: archive.zip suministrado por el usuario. El README original declara CC BY 4.0 y el origen https://universe.roboflow.com/salman-7sifm/scoliosis-ceson-nmdrq/dataset/1 . Conservar la atribucion al autor; no es una validacion independiente de procedencia o licencia.\n\n## Uso\nExtraer el ZIP y usar la ruta absoluta a data.yaml para entrenamiento YOLO. Las rutas train/val/test son relativas a la carpeta del YAML (Ultralytics). No volver a repartir aleatoriamente imagen por imagen.\n\n## Metodo\n- Se verifican decodificacion de imagenes, clases y coordenadas.\n- Tres anotaciones poligonales se convierten geometricamente a sus cajas envolventes, conservando la clase. No eran datos corruptos, sino formatos mezclados.\n- Se recortan geometricamente al borde de la imagen las cajas que lo exceden; cada cambio queda registrado.\n- Se agrupan variantes por nombre anterior a .rf., bytes identicos, pixeles RGB identicos y reflejos horizontales exactos. Estas claves se unen transitivamente. No se consideran identificadores de paciente.\n- Imagenes de pixeles identicos con etiquetas incompatibles (clase, cantidad o coordenadas con diferencia mayor a 0.01) quedan en cuarentena logica: el manifest permite recuperarlas del ZIP original. No se inventa la etiqueta correcta.\n- Se conserva una copia por imagen RGB identica. Se conservan otras variantes aumentadas, siempre dentro del mismo grupo.\n- Se reconstruye una particion aproximada 70% entrenamiento, 25% prueba y 5% validacion, equilibrando presencia de clases y numero de imagenes, sin dividir grupos. Semilla base 20261114, busqueda determinista de 120 candidatos.\n- audit.json contiene cantidades y comprobaciones; manifest.csv conserva trazabilidad de TODOS los archivos originales; changes.json registra conversiones y cuarentena.\n\n## Limites\nNo hay identificadores de paciente verificados. Pueden existir imagenes relacionadas con nombres diferentes o aumentos no detectados. Las particiones de evaluacion pueden conservar aumentos preexistentes; no es posible reconocer con certeza todos los originales. Para una evaluacion definitiva, obtener originales sin aumentos e identificadores de paciente, dividir por paciente y aumentar solo train. Las etiquetas clinicas y las cajas necesitan revision experta. No se garantiza balance exacto: prima la integridad de los grupos.\n\n## Reproducir\nPython 3 con numpy y Pillow. Ejecutar: python prepare_dataset.py --input RUTA_ARCHIVE_ZIP --output CARPETA_NUEVA\n\n'''+json.dumps(report,indent=2)
 (out/'README.md').write_text(readme,encoding='utf-8')
 for n in z.namelist():
  if 'README.' in n: (out/('original_'+Path(n).name)).write_bytes(z.read(n))
 shutil.copy2(__file__,out/'prepare_dataset.py')
 print(json.dumps(report,indent=2),flush=True)
 archive=shutil.make_archive(str(out.parent/'scoliosis_depurado'),'zip',out.parent,out.name)
 print('ZIP',archive,flush=True)
if __name__=='__main__': main()
