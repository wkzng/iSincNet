import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset


from training.trainer import BaseTrainer, TrainConfig, STFT




class Trainer(BaseTrainer):
    def __init__(self, model:torch.nn.Module, train_set:Dataset, val_set:Dataset, config:TrainConfig) -> None:
        super().__init__(
            model=model, 
            train_set=train_set, 
            val_set=val_set, 
            config=config,
            current_script_path=os.path.abspath(__file__)
        )
        self.scaler = torch.amp.GradScaler(device=self.device)
        self.amp_enabled = self.device == "cuda"


    def train_one_epoch(self, current_epoch:int):
        """evaluate the model on the train dataset"""
        self.model.train()
        n_batches = len(self.train_loader)
        lr = self.scheduler.get_lr()[0]
        progress_bar = tqdm(enumerate(self.train_loader), ncols=200, total=n_batches, disable=False)

        for b_idx, batch in progress_bar:
            #transforms = self.audio_augmenter.get_random_transforms(k=3)
            transforms = [self.audio_augmenter.transforms[0]]
            self.optimizer.zero_grad()
            
            #with torch.amp.autocast(device_type="xpu", enabled=self.amp_enabled, dtype=torch.float16):
            with torch.amp.autocast(device_type=str(self.device), enabled=False, dtype=torch.float16):

                #compute local losses
                local_losses = {}
                waveforms = batch["waveform"].to(self.device)

                waveforms[0] *= 0
                if 2 * np.random.random() < 1:
                    waveforms = torch.flip(waveforms, dims=[1])
                if 2 * np.random.random() < 1:
                    waveforms = - waveforms

                for T in transforms:
                    transformed_wav = T(waveforms)
                    reconstructed_wav = self.model(transformed_wav)

                    transformed_stft = self.stft.compute_log1p_magnitude(transformed_wav)
                    reconstructed_stft = self.stft.compute_log1p_magnitude(reconstructed_wav)

                    loss_l1 = F.l1_loss(transformed_wav, reconstructed_wav)
                    loss_l2 = F.mse_loss(transformed_wav, reconstructed_wav)
                    loss_msl = F.l1_loss(transformed_stft, reconstructed_stft) 

                    if torch.isfinite(loss_l1):
                        local_losses["tL1"] = local_losses.get("tL1", 0) + loss_l1
                    if torch.isfinite(loss_l2):
                        local_losses["tL2"] = local_losses.get("tL2", 0) + loss_l2
                    if torch.isfinite(loss_msl):
                        local_losses["msl"] = local_losses.get("msl", 0) + loss_msl

                #aggregate local losses into global train loss
                train_loss = 0
                for _, loss in local_losses.items():
                    train_loss += loss / len(transforms)

            #skip the backprop step if the loss is ill-defined
            if not isinstance(train_loss, torch.Tensor) or not torch.isfinite(loss):
                print("Skipping....")
                continue
            
            #perform backprop with AMP scaler
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            #tensorboard logs
            for k, v in local_losses.items():
               self.writer.add_scalar(k, v, self.n_steps)

            #training metrics log for stdout
            message = f"{current_epoch}/{self.config.n_epoch} | lr={lr:0.3} "
            for k, v in local_losses.items():
                message += " | {}={:0.6}".format(k, float(v))
            progress_bar.set_description(message)
            #print(message, end='\r')
            
            #checkpoint snapshot
            if self.n_steps>0 and self.n_steps % 100 == 0:
               self.save_checkpoint(stats={}, current_epoch=current_epoch, n_steps=self.n_steps)

            #increment the step tracker
            self.n_steps += 1



    def train(self):
        """train the model on the entire dataset"""
        checkpoint = self.load_checkpoint()
        start_epoch = checkpoint.get("epoch", 0)
        self.n_steps = checkpoint.get("n_steps", 0)

        for current_epoch in range(start_epoch, self.config.n_epoch):
            self.train_one_epoch(current_epoch)
            self.save_checkpoint(stats={}, current_epoch=current_epoch, n_steps=self.n_steps)
            self.scheduler.step()
        self.writer.close()




if __name__ =="__main__":
    from datasets.configs import GTZANConfig, EDMConfig, LegacyGenresConfig
    from datasets.dataset import ChunkDataset
    from model import SincNet

    model = SincNet()
    dataset_config = GTZANConfig()
    train_config = TrainConfig(**{
        "batch_size": 8,
        "n_epoch": 500,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "training_id": dataset_config.id
    })


    datasets = {
        mode:ChunkDataset(
            parquet_file_path=dataset_config.parquet,
            model_config_file_path=dataset_config.config,
            h5_file_path = dataset_config.hdf5,
            mode=mode,
        )
        for mode in ["train", "test"]
    }
    
    trainer = Trainer(
        model=model, 
        train_set=datasets["train"], 
        val_set=datasets["test"],
        config=train_config
    )
    trainer.train()