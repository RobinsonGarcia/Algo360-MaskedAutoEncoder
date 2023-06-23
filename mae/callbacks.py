from pytorch_lightning.callbacks import Callback

import numpy as np

import wandb
import torch
import os

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def run_one_image(x, model,to_wandb):
    

    # make it a batch-like
    x = x.unsqueeze(dim=0)

    #x = torch.einsum('nhwc->nchw', x)


    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    
    to_rgb = lambda x:torch.clip((x * imagenet_std + imagenet_mean) * 255, 0, 255).int()

    r = lambda x:wandb.Image(to_rgb(x)) if to_wandb else to_rgb(x)
    
    return [
            ("original",r(x)),
    ("masked",r(im_masked)),
    ("reconstruction",r(y)),
    ("reconstruction + visible",r(im_paste))
    ]

class mae_Callback(Callback):
    def __init__(self,to_wandb,save_samples_to=None):
        super().__init__()
        self.to_wandb = to_wandb
        self.save_samples_to = save_samples_to

    def on_predict_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pass
           
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.generate_imgs('train',trainer, pl_module, outputs, batch, batch_idx, unused=0)
        pass

    def on_test_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.generate_imgs('test',trainer, pl_module, outputs, batch, batch_idx, unused=0)
        pass

    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self.generate_imgs('val',trainer, pl_module, outputs, batch, batch_idx, unused=0)
        pass  

    def generate_imgs(self,mode,trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if batch_idx%100 == 0:
            N = batch.shape[0]
            data = []
            for n in range(4):
                output_batch = run_one_image(batch[n], pl_module.model, self.to_wandb)
                output_batch = dict(output_batch)
                data.append(list(output_batch.values()))
                columns = list(output_batch.keys())

                if isinstance(self.save_samples_to,str):
                    np.savez_compressed(os.path.join(self.save_samples_to,f'{mode}_{n}_{batch_idx}.npz'))

            if self.to_wandb: pl_module.logger.log_table(
                key=f'Sample result (from the {mode} set)',
                columns=columns,
                data=data)  
        pass
