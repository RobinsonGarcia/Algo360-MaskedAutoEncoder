

import os
import PIL

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule

def load_files(mode,path2csv):

    df = pd.read_csv(path2csv)
    mask = df['mode'].map(lambda x: True if x == mode else False )
    files = df[mask].files.tolist()
    return files

class mae_Dataset(Dataset):
        
    def __init__(
            self,
            mode,
            transform=None,
            preprocessing_fn=None,
            **cfg):
        
        super().__init__()

        print(cfg)
        
        self.cfg = cfg
        self.files_fps = load_files(mode,cfg['path2csv'])
        self.transform = transform
        self.preprocessing_fn = preprocessing_fn
    
    def __len__(self):
        return len(self.files_fps)

    def __getitem__(self, idx):
        
        filename = self.files_fps[idx]
        
        rgb_image = Image.fromarray(np.load(filename,allow_pickle=True)['face'])
        
        batch = self.transform(rgb_image)

        
  
        return batch



def build_dataset(mode, hparams):
    transform = build_transform(True if mode=='train' else False, hparams)

    dataset = mae_Dataset(mode,transform=transform,**hparams)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class mae_DataModule(LightningDataModule):
      
    def __init__(self,
    batch_size,
    num_trainloader_workers,
    num_validloader_workers,
    input_size,
    color_jitter,
    remode,
    recount,
    path2csv,
    aa,
    reprob,
    **cfg):
        super().__init__()

        self.save_hyperparameters()
        print(self.hparams)

    def setup(self,stage=None,mode=None):
        if stage=='fit':

            self.corr_train = build_dataset('train', self.hparams)
            self.corr_val = build_dataset('val', self.hparams)
        
        if stage=='test':
            self.corr_test = build_dataset('test', self.hparams)
        
        if stage=='predict':
            self.corr_predict = None
            print('running inference on {} images'.format(len(self.corr_predict)))

    def train_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_train,num_replicas=self.num_replicas,shuffle=True,) if 'dp' in self.cfg['strategy'] else None    
        #except:
        #    sampler=None
        return DataLoader(self.corr_train,batch_size=self.hparams.batch_size,drop_last=True , num_workers=self.hparams['num_trainloader_workers'],pin_memory=True,prefetch_factor =5,persistent_workers=True)

    def val_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_val,num_replicas=self.num_replicas) if 'dp' in self.cfg['strategy'] else None   
        #except:
        #    sampler=None
        return DataLoader(self.corr_val,  batch_size=self.hparams.batch_size,drop_last=True , num_workers=self.hparams['num_validloader_workers'],pin_memory=True)
        
    def test_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_test,num_replicas=self.num_replicas) if 'dp' in self.cfg['strategy'] else None   
        #except:
        #    sampler=None
        ##### HACK for scale shift compute, change corr_test
        return DataLoader(self.corr_train,  batch_size=self.self.hparams.batch_size,drop_last=True , num_workers=self.hparams['num_validloader_workers'],pin_memory=True)
    
    def predict_dataloader(self):
        #try:
        #    sampler = DistributedSampler(self.corr_test,num_replicas=self.num_replicas) if 'dp' in self.cfg['strategy'] else None   
        #except:
        #    sampler=None
        ##### HACK for scale shift compute, change corr_test
        #return DataLoader(self.corr_predict,  batch_size=self.batch_size,drop_last=False , num_workers=self.hparams['num_validloader_workers'],pin_memory=True)
        pass
    
    
