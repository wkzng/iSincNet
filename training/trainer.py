import json, os
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from training.utils.general import set_random_seed, custom_load_state_dict
from training.utils.audio_augmentations import AudioAugmenter
from training.utils.stft import TorchSTFT



@dataclass
class TrainConfig:
    n_epoch : int = 50
    training_id : str = "gtzan"
    batch_size : int = 4
    train_workers : int = 8
    val_workers : int = 2
    sample_rate : int = 22050
    learning_rate : int = 1e-4
    weight_decay : int = 1e-6
    tnse_log_steps_freq : int = 100
    ckpt_freq : int = 50
    random_seed : int = 2024
    clip_grad: float = 2.0



@dataclass
class STFT:
    sample_rate : int = 22050
    n_fft : int = 1024
    f_max :int = 8000
    win_length : int = 1001
    n_mels : int = 160



class BaseTrainer:
    def __init__(self, 
            model:torch.nn.Module, 
            train_set:Dataset, 
            val_set:Dataset, 
            config:TrainConfig,
            current_script_path:str
        ) -> None:
        self.config = config
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.n_epoch = config.n_epoch
        self.ckpt_freq = config.ckpt_freq
        self.stft_config = STFT(sample_rate=config.sample_rate)
        self.current_script_path = current_script_path 

        set_random_seed(config.random_seed)
        self.initialise_training_dirs()
        self.initialise_device()
        self.intialise_dataloaders()
        self.initialise_tensorboard_writer()
        self.initialise_audio_augmentation()
        self.initialise_optimizer_and_scheduler()


    def initialise_training_dirs(self):
        """Generate path to all logs and checkpoint dirs"""
        current_datetime = datetime.now()
        current_time = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
        current_script_dir = os.path.dirname(self.current_script_path)

        self.logs_dir = os.path.join(current_script_dir, "trainings", self.config.training_id, "logs")
        self.checkpoint_dir = os.path.join(current_script_dir,"trainings", self.config.training_id, "ckpt")
        self.tensorboard_dir = os.path.join(current_script_dir,"trainings", self.config.training_id, f"tensorboard/{current_time}")

        for directory in  [self.logs_dir, self.checkpoint_dir, self.tensorboard_dir]:
            os.makedirs(directory, exist_ok=True)


    def initialise_device(self):
        """set the training device to cpu or cuda depending on available device"""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Nbr of CUDA devices: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        elif torch.xpu.is_available():
            device = "xpu"
            print(f"Nbr of XPU devices: {torch.xpu.device_count()}")
            print(f"Current XPU device: {torch.xpu.current_device()}")
            
            print("Overriding XPU device to CPU")
            device = "cpu"
        else:
            device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)
        self.stft = TorchSTFT(**asdict(self.stft_config)).to(device=self.device)
        print(f"Training device: {self.device}")


    def intialise_dataloaders(self):
        """initialise the trainloader and testloader"""
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.batch_size, 
            num_workers=self.config.train_workers, 
            pin_memory=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config.batch_size, 
            num_workers=self.config.val_workers, 
            pin_memory=True, 
            shuffle=False
        )
    

    def initialise_tensorboard_writer(self):
        """initise the tensorbord log writer in order to keep track of loss, accuracy and TNSE"""
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.writer.add_text("arguments", json.dumps(asdict(self.config)))


    def initialise_audio_augmentation(self):
        """initise the CPU and GPU compatible audio augmenation (reverb, pitch shift, etc...) module """
        self.audio_augmenter = AudioAugmenter(sr=self.config.sample_rate)


    def initialise_optimizer_and_scheduler(self):
        """intialiset the optimiser and the scheduler"""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        #Learning rate scheduler
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4,  T_mult=1, eta_min=1e-4)
        #scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=4, mode="triangular")
        self.scheduler = StepLR(self.optimizer, step_size=4, gamma=0.95)


    def load_checkpoint(self) -> dict:
        """initialise the checkpoint directory and load pretrained_weights if possible"""
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model.name}.ckpt")
        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                #self.model.load_state_dict(checkpoint.pop("state_dict"))
                custom_load_state_dict(self.model, checkpoint.pop("state_dict"))
                #self.optimizer.load_state_dict(checkpoint.pop("optimizer_state"))
                #self.scheduler.load_state_dict(checkpoint.pop("scheduler_state"))
                print(f"Loaded checkpoint from {ckpt_path}...")
                return checkpoint
            except:
                print(f"Failure while loading checkpoint from {ckpt_path}...")
        else:
            print(f"Checkpoint not found at:{ckpt_path}...")
        return {}
    

    def save_checkpoint(self, stats:dict, current_epoch:int, n_steps:int):
        """save the current training parameters as checkpoint"""
        state = {
            "state_dict":self.model.state_dict(), 
            "optimizer_state":self.optimizer.state_dict(), 
            "scheduler_state":self.scheduler.state_dict(),
            "epoch": current_epoch,
            "n_steps": n_steps
        }
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model.name}.ckpt")
        torch.save({**stats, **state}, ckpt_path)


    def evaluate(self, current_epoch:int):
        """evaluate the model on the train dataset"""
        pass


    def train_one_epoch(self, current_epoch:int):
        """evaluate the model on the train dataset"""
        pass


    def train(self):
        """train the model on the entire dataset"""
        pass

