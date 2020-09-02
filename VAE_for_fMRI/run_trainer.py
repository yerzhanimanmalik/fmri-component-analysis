
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.arguments import args
from src.Trainer.ae_base import AutoencoderTrainer
from src.utils import fMRIDataset
from src.vae.vanilla import VanillaVAE

num_epochs = args['num_epochs']
gpus = args['gpus']
dataset_path = args['dataset_path']
input_size = args['input_size']

transform = transforms.Compose([transforms.Resize(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset_train = fMRIDataset(dataset_path+'train', transform=transform)
dataloader_train = DataLoader(dataset_train, num_workers=4, batch_size=128)

dataset_val = fMRIDataset(dataset_path+'val', transform=transform)
dataloader_val = DataLoader(dataset_val, num_workers=4, batch_size=128)

vae = VanillaVAE(z_dim=args['z_dim'])
ae_trainer = AutoencoderTrainer(vae)

trainer = pl.Trainer(max_epochs=num_epochs,
                     gpus=gpus)

trainer.fit(model=ae_trainer,
            train_dataloader=dataloader_train,
            val_dataloaders=dataloader_val)