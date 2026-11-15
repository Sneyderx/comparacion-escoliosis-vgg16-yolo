"""Shared image transforms, data loaders and model adapters for both networks."""
import csv, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image,ImageOps

NAMES=['1-derece','2-derece','3-derece','saglikli']
class Radiographs(Dataset):
 def __init__(self,root,manifest,split,cfg,seed,limit=None):
  self.root=Path(root); self.cfg=cfg; self.seed=seed; self.epoch=0; self.augment=split=='train'
  with open(manifest,encoding='utf-8-sig') as f: self.rows=[r for r in csv.DictReader(f) if r['split']==split]
  if limit:
   # Include each class for a useful smoke test, avoiding a single-class prefix.
   self.rows=[r for c in range(4) for r in [x for x in self.rows if int(x['class_id'])==c][:limit]]
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['relative_path']) as source: im=source.convert('RGB')
  n=self.cfg['image_size']; im=ImageOps.pad(im,(n,n),method=Image.Resampling.BILINEAR,color=(0,0,0),centering=(.5,.5))
  if self.augment:
   rng=random.Random(self.seed*1000003+self.epoch*len(self)+i)
   if rng.random()<self.cfg['horizontal_flip_probability']: im=ImageOps.mirror(im)
   im=TF.rotate(im,rng.uniform(-self.cfg['rotation_degrees'],self.cfg['rotation_degrees']),interpolation=InterpolationMode.BILINEAR,fill=0)
  return TF.to_tensor(im),int(r['class_id']),r['relative_path'],r['group']
class Network(nn.Module):
 def __init__(self,name,cfg,pretrained):
  super().__init__(); self.name=name
  if name=='vgg16':
   from torchvision.models import vgg16,VGG16_Weights
   self.net=vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
   self.net.classifier[-1]=nn.Linear(self.net.classifier[-1].in_features,4)
   for layer in self.net.modules():
    if isinstance(layer,nn.Dropout): layer.p=cfg['dropout']
   self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1)); self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
  elif name=='yolo26n_cls':
   from ultralytics import YOLO
   from ultralytics.nn.tasks import ClassificationModel
   self.net=YOLO('yolo26n-cls.pt' if pretrained else 'yolo26n-cls.yaml',task='classify').model.float()
   ClassificationModel.reshape_outputs(self.net,4)
   self.net.names=dict(enumerate(NAMES))
   self.net.nc=4
   for par in self.net.parameters(): par.requires_grad_(True)
   for layer in self.net.modules():
    if isinstance(layer,nn.Dropout): layer.p=cfg['dropout']
   self.net.model[-1].export=False
  else: raise ValueError(name)
 def forward(self,x):
  if self.name=='vgg16': return self.net((x-self.mean)/self.std)
  out=self.net(x)
  # Official Classify: logits during train; (probabilities, logits) in eval.
  if isinstance(out,tuple): out=out[1]
  if not isinstance(out,torch.Tensor) or out.ndim!=2 or out.shape[1]!=4: raise RuntimeError('Unexpected YOLO classification API; check installed version')
  return out
