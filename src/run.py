"""One isolated training or evaluation run; test is never read by train()."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,platform,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report,roc_auc_score
from common import Network,Radiographs,NAMES

def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def signature(cfg,manifest): return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+Path(manifest).read_bytes()).hexdigest()
def synchronize(device):
 if device.type=='cuda': torch.cuda.synchronize(device)
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True)
def metrics(y,prob):
 pred=prob.argmax(1); p,r,f,_=precision_recall_fscore_support(y,pred,labels=list(range(4)),average='macro',zero_division=0)
 result=dict(accuracy=float(accuracy_score(y,pred)),precision_macro=float(p),recall_macro=float(r),f1_macro=float(f))
 cm=confusion_matrix(y,pred,labels=list(range(4))); total=cm.sum(); specs=[]
 for k in range(4):
  tn=total-cm[k,:].sum()-cm[:,k].sum()+cm[k,k]; fp=cm[:,k].sum()-cm[k,k]; specs.append(float(tn/(tn+fp)) if tn+fp else 0.)
 result['specificity_macro']=float(np.mean(specs))
 try: result['auroc_macro_ovr']=float(roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=list(range(4))))
 except ValueError: result['auroc_macro_ovr']=None
 return result

def loader(ds,cfg,shuffle=False,epoch=0):
 generator=torch.Generator().manual_seed(ds.seed+epoch)
 return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=cfg['num_workers'],generator=generator,pin_memory=torch.cuda.is_available(),drop_last=False)
@torch.no_grad()
def predict(model,dl,device):
 model.eval(); ys=[]; ps=[]; paths=[]; groups=[]; loss=0.; criterion=nn.CrossEntropyLoss()
 for x,y,p,g in dl:
  x=x.to(device); y=y.to(device); logits=model(x); loss+=float(criterion(logits,y))*len(y)
  ys.extend(y.cpu().tolist()); ps.extend(logits.softmax(1).cpu().numpy()); paths.extend(p); groups.extend(g)
 return np.asarray(ys),np.asarray(ps),paths,groups,loss/len(ys)

def train(a,cfg,folder,device):
 sig=signature(cfg,a.manifest); done=folder/'training_done.json'
 if done.exists():
  assert json.loads(done.read_text())['signature']==sig, 'Configuration/data changed: use a fresh results directory'
  print('Already trained:',folder); return
 if (folder/'best.pt').exists(): raise RuntimeError(f'Partial run found: {folder}. Choose a fresh results root; do not combine partial and complete runs.')
 seed_all(a.seed); model=Network(a.model,cfg,cfg['pretrained']).to(device)
 # Assertions cover class-head reshaping and both training/evaluation forward outputs.
 check=torch.zeros(2,3,cfg['image_size'],cfg['image_size'],device=device)
 model.eval()
 with torch.no_grad(): assert model(check).shape==(2,4)
 del check
 tr=Radiographs(a.dataset,a.manifest,'train',cfg,a.seed,1 if a.smoke else None)
 va=Radiographs(a.dataset,a.manifest,'val',cfg,a.seed,1 if a.smoke else None)
 optimizer=torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
 criterion=nn.CrossEntropyLoss(); best=(-1.,-float('inf')); waiting=0; history=[]
 if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
 synchronize(device); started=time.perf_counter()
 for epoch in range(1,cfg['max_epochs']+1):
  model.train(); tr.epoch=epoch; train_loss=0.; count=0
  for x,y,_,_ in loader(tr,cfg,True,epoch):
   x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step(); train_loss+=float(loss.detach())*len(y); count+=len(y)
  y,prob,_,_,val_loss=predict(model,loader(va,cfg),device); met=metrics(y,prob)
  row=dict(epoch=epoch,train_loss=train_loss/count,val_loss=val_loss,val_f1_macro=met['f1_macro']); history.append(row)
  with (folder/'history.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,row.keys()); w.writeheader(); w.writerows(history)
  score=(met['f1_macro'],-val_loss)
  if score>best:
   best=score; waiting=0; torch.save(dict(state_dict=model.state_dict(),config=cfg,model=a.model,seed=a.seed,epoch=epoch,signature=sig),folder/'best.pt')
  else: waiting+=1
  print(a.model,a.seed,row,flush=True)
  if waiting>=cfg['patience']: break
 synchronize(device)
 environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,precision='FP32',deterministic_algorithms='warn_only; inspect warnings',parameter_count=sum(p.numel() for p in model.parameters()))
 dump(folder/'training_done.json',dict(model=a.model,seed=a.seed,signature=sig,config=cfg,final_epoch=epoch,training_and_validation_seconds=time.perf_counter()-started,peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,environment=environment))

def gradcam(model,ds,indices,folder,device):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import torch.nn.functional as F
 activation=[]
 def hook(_,__,out): activation.clear(); activation.append(out); out.retain_grad()
 handle=model.net.features[-1].register_forward_hook(hook)
 model.eval()
 for j in indices:
  x,y,path,_=ds[j]; x=x.unsqueeze(0).to(device); model.zero_grad(set_to_none=True); logits=model(x); pred=int(logits.argmax(1)); logits[0,pred].backward()
  act=activation[0]; weights=act.grad.mean((2,3),keepdim=True); cam=(weights*act).sum(1,keepdim=True).relu(); cam=F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy(); cam=(cam-cam.min())/(np.ptp(cam)+1e-9)
  fig,ax=plt.subplots(figsize=(4,5)); ax.imshow(x[0].detach().cpu().permute(1,2,0).numpy()); ax.imshow(cam,cmap='jet',alpha=.40,vmin=0,vmax=1); ax.set_title(f'True: {NAMES[y]} | predicted: {NAMES[pred]}\nGrad-CAM for predicted class',fontsize=9); ax.axis('off'); fig.tight_layout(); fig.savefig(folder/f'gradcam_{j:04d}.png',dpi=160); plt.close(fig)
 handle.remove()

def evaluate(a,cfg,folder,device):
 checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
 assert checkpoint['signature']==signature(cfg,a.manifest)
 seed_all(a.seed); model=Network(a.model,cfg,False); model.load_state_dict(checkpoint['state_dict']); model=model.to(device).eval()
 test=Radiographs(a.dataset,a.manifest,'test',cfg,a.seed,1 if a.smoke else None)
 y,prob,paths,groups,_=predict(model,loader(test,cfg),device); result=metrics(y,prob)
 rows=[dict(path=p,group=g,true_class=int(t),predicted_class=int(v.argmax()),confidence=float(v.max()),**{f'p{k}':float(v[k]) for k in range(4)}) for p,g,t,v in zip(paths,groups,y,prob)]
 import pandas as pd
 pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
 pd.DataFrame([r for r in rows if r['true_class']!=r['predicted_class']]).to_csv(folder/'errors.csv',index=False)
 dump(folder/'classification_report.json',classification_report(y,prob.argmax(1),labels=list(range(4)),target_names=NAMES,output_dict=True,zero_division=0))
 cm=confusion_matrix(y,prob.argmax(1),labels=list(range(4))); np.savetxt(folder/'confusion_matrix.csv',cm,delimiter=',',fmt='%d')
 # Network-only FP32, batch=1, preloaded input; synchronize GPU for each timing.
 x=test[0][0].unsqueeze(0).to(device); durations=[]
 with torch.inference_mode():
  for _ in range(cfg['latency_warmup']): model(x)
  synchronize(device)
  for _ in range(cfg['latency_repetitions']):
   synchronize(device); start=time.perf_counter(); model(x); synchronize(device); durations.append((time.perf_counter()-start)*1000)
 training=json.loads((folder/'training_done.json').read_text())
 result.update(model=a.model,seed=a.seed,best_epoch=checkpoint['epoch'],final_epoch=training['final_epoch'],training_and_validation_seconds=training['training_and_validation_seconds'],latency_mean_ms=float(np.mean(durations)),latency_median_ms=float(np.median(durations)),latency_p95_ms=float(np.percentile(durations,95)),parameter_count=training['environment']['parameter_count'],peak_gpu_memory_mb=training['peak_gpu_memory_mb'],training_device=training['environment']['gpu'] or 'cpu',device=torch.cuda.get_device_name(device) if device.type=='cuda' else str(device),signature=checkpoint['signature'])
 dump(folder/'metrics.json',result); pd.DataFrame([result]).to_csv(folder/'metrics.csv',index=False)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from sklearn.metrics import ConfusionMatrixDisplay
 fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm,display_labels=NAMES).plot(ax=ax,cmap='Blues',colorbar=False); fig.tight_layout(); fig.savefig(folder/'confusion_matrix.png',dpi=160); plt.close(fig)
 if a.model=='vgg16' and a.seed==cfg['seeds'][0]:
  errors=np.where(y!=prob.argmax(1))[0].tolist(); correct=np.where(y==prob.argmax(1))[0].tolist(); gradcam(model,test,errors[:6]+correct[:2],folder,device)
 print(json.dumps(result,indent=2))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['train','evaluate'],required=True); p.add_argument('--model',choices=['vgg16','yolo26n_cls'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--dataset',required=True); p.add_argument('--manifest',default='data/classification.csv'); p.add_argument('--config',default='config.json'); p.add_argument('--results',default='results'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
 if a.smoke: cfg.update(max_epochs=1,batch_size=4,num_workers=0,image_size=64,pretrained=False,latency_warmup=1,latency_repetitions=2)
 folder=Path(a.results)/('smoke' if a.smoke else 'experiment')/a.model/f'seed_{a.seed}'; folder.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if not a.smoke and device.type!='cuda': raise RuntimeError('Enable a GPU runtime for the full experiment')
 (train if a.mode=='train' else evaluate)(a,cfg,folder,device)
if __name__=='__main__': main()
