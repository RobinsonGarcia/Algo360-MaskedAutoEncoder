from pytorch_lightning import LightningModule
import torch

import timm.optim.optim_factory as optim_factory

import mae.models_mae as models_mae

class pl_MaskedAutoEncoder(LightningModule):


    def __init__(self,
    model,
    img_size,
    lr,
    weight_decay,
    **cfg):
        super().__init__()
        #self.save_hyperparameters(ignore=['criterion','optimizer'])
        self.save_hyperparameters()

        self.model = models_mae.__dict__[self.hparams.model](**{'img_size':self.hparams.img_size})


    def forward(self,x,mask_ratio=0.75):
        return self.model(x,mask_ratio=mask_ratio)
        
    def configure_optimizers(self):
        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.hparams.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr, betas=(0.9, 0.95))
        print(optimizer)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= 10 , T_mult = 1 , eta_min =1e-7)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


    def training_step(self, batch, batch_idx):
        loss,pred,mask = self(batch)
    
        
        return {'loss':loss,'y_hat':pred,'mask':mask}

    def validation_step(self, batch, batch_idx):
        val_loss,pred,mask = self(batch)
    
        
        return {'val_loss':val_loss,'y_hat':pred,'mask':mask}

    def test_step(self, batch, batch_idx):
    

        pass

    def predict_step(self, batch, batch_idx):

        x = batch['processed_input'][0]
        latent, mask, ids_restore = self.forward_encoder(batch, 0)
        

        return {'latent':latent,'mask':mask,'ids_restore':ids_restore}




