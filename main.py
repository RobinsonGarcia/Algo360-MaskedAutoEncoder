import os
import wandb

#os.environ['SINGULARITYENV_WANDB_CACHE_DIR']='/nethome/algo360/mestrado/monocular-depth-estimation/.cache'
 
#os.environ['WANDB_CACHE_DIR']='/nethome/algo360/mestrado/monocular-depth-estimation/.cache'

#os.environ['WANDB_DATA_DIR']='/nethome/algo360/mestrado/monocular-depth-estimation/.wandb'
# main.py
from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from mae.pl_datamodule import mae_DataModule
from mae.pl_model import pl_MaskedAutoEncoder

def cli_main():
    config_filename = 'tmp.yaml'#config_run{}.yaml'.format(os.environ['SLURM_JOB_ID'])
    cli = LightningCLI(pl_MaskedAutoEncoder, mae_DataModule,save_config_kwargs={"config_filename":config_filename,"overwrite": False})
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    wandb.finish()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
