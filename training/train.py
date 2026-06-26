import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


from training.trainer import BaseTrainer, TrainConfig
from training.utils.losses import MultiResolutionSTFTLoss



def calculate_si_sdr(references: torch.Tensor, estimates: torch.Tensor) -> torch.Tensor:
    """
    Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        references (torch.Tensor): The ground truth waveforms (B, T).
        estimates (torch.Tensor): The reconstructed waveforms (B, T).

    Returns:
        torch.Tensor: The mean SI-SDR score across the batch (scalar, in dB).
    """
    # Ensure inputs are 2D (B, T) - remove the channel dimension if present
    if references.ndim == 3:
        references = references.squeeze(1)
    if estimates.ndim == 3:
        estimates = estimates.squeeze(1)

    # 1. Zero-mean the signals
    references = references - torch.mean(references, dim=-1, keepdim=True)
    estimates = estimates - torch.mean(estimates, dim=-1, keepdim=True)

    # 2. Find the optimal scaling factor (alpha) for the projection
    # alpha = <estimates, references> / ||references||^2
    # s_target = alpha * references

    # Calculate projection of estimates onto references
    s_target_num = torch.sum(estimates * references, dim=-1, keepdim=True)
    s_target_den = torch.sum(references ** 2, dim=-1, keepdim=True)

    # Handle the unlikely case of silent reference to avoid division by zero
    s_target_den = torch.where(s_target_den == 0, torch.full_like(s_target_den, 1e-8), s_target_den)

    alpha = s_target_num / s_target_den
    s_target = alpha * references

    # 3. Calculate the noise (e_noise)
    # e_noise = estimates - s_target
    e_noise = estimates - s_target

    # 4. Calculate SDR (in linear scale)
    # SDR = ||s_target||^2 / ||e_noise||^2
    s_target_power = torch.sum(s_target ** 2, dim=-1)
    e_noise_power = torch.sum(e_noise ** 2, dim=-1)

    # Handle the unlikely case of zero noise power
    e_noise_power = torch.where(e_noise_power == 0, torch.full_like(e_noise_power, 1e-8), e_noise_power)

    # 5. Convert to Decibels and return the mean
    si_sdr = 10 * torch.log10(s_target_power / e_noise_power)

    # SI-SDR should be returned as a scalar average over the batch
    return torch.mean(si_sdr)




class SincNetTrainer(BaseTrainer):
    """Autoencoder trainer for a plain SincNet (waveform -> spectrogram -> waveform)."""

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
        # band-normalised multi-resolution STFT loss: weights high frequencies as much as low ones,
        # so the decoder stops leaving the high band smeared (the frame-rate stripes).
        self.mrstft = MultiResolutionSTFTLoss().to(self.device)


    def energy(self, x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
        """Calculates the energy (sum of squares) of a tensor along a dimension."""
        return torch.mean(x**2, dim=dim, keepdim=keepdim)


    def model_step(self, waveforms:torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Run the model and return (reconstructed_waveform, extra_losses).

        Base SincNet returns a waveform tensor and no extra losses; subclasses (e.g. the adapter)
        override this to unpack a richer output and contribute extra terms to the objective."""
        return self.model(waveforms), {}


    @torch.no_grad()
    def evaluate(self, current_epoch:int) -> dict[str, float]:
        """evaluate the model on the train dataset"""
        self.model.eval()
        n_batches = len(self.val_loader)
        if n_batches == 0:
            return {}

        progress_bar = tqdm(enumerate(self.val_loader), ncols=200, total=n_batches, disable=False, leave=False)
        scores = {
            "L1": 0.0,
            "L2": 0.0,
            "stft": 0.0,
            "mrstft": 0.0,
            "si_sdr": 0.0,
        }

        for b_idx, batch in progress_bar:
            waveforms = batch["waveform"].to(self.device).squeeze(1)
            reconstructed_wav, extra_losses = self.model_step(waveforms)

            scores["L1"] += F.l1_loss(waveforms, reconstructed_wav)
            scores["L2"] += F.mse_loss(waveforms, reconstructed_wav)
            scores["si_sdr"] += calculate_si_sdr(waveforms, reconstructed_wav)

            stft_targ = self.stft.compute_log1p_magnitude(waveforms)
            stft_pred = self.stft.compute_log1p_magnitude(reconstructed_wav)
            scores["stft"] += F.l1_loss(stft_targ, stft_pred)
            scores["mrstft"] += self.mrstft(reconstructed_wav.float(), waveforms.float())
            for k, v in extra_losses.items():
                scores[k] = scores.get(k, 0.0) + v

        scores = {k: v / n_batches for k, v in scores.items()}
        scores["log-L1"] = torch.log(scores.pop("L1"))
        scores["log-L2"] = torch.log(scores.pop("L2"))
        scores["log-stft"] = torch.log(scores.pop("stft"))

        #tensorboard logs
        for k, v in scores.items():
            self.writer.add_scalar(f"Eval:{k}", v, self.n_steps)

        #training metrics log for stdout
        message = f"Eval | {current_epoch}/{self.config.n_epoch}"
        for k, v in scores.items():
            message += " | {}={:0.6}".format(k, float(v))
        print(message)
        return scores


    def train_one_epoch(self, current_epoch:int):
        """evaluate the model on the train dataset"""
        self.model.train()
        n_batches = len(self.train_loader)
        lr = self.scheduler.get_lr()[0]
        progress_bar = tqdm(enumerate(self.train_loader), ncols=200, total=n_batches, disable=False, leave=False)

        for b_idx, batch in progress_bar:
            #transforms = self.audio_augmenter.get_random_transforms(k=3)
            transforms = [self.audio_augmenter.transforms[0]]
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=str(self.device), enabled=self.amp_enabled, dtype=torch.float16):
                #compute local losses
                local_losses = {}
                waveforms = batch["waveform"].to(self.device)

                if 2 * np.random.random() < 1:
                    waveforms = torch.flip(waveforms, dims=[-1])
                if 2 * np.random.random() < 1:
                    waveforms = - waveforms

                #print("\n================")
                for T in transforms:
                    transformed_wav = T(waveforms).squeeze(1)
                    reconstructed_wav, extra_losses = self.model_step(transformed_wav)

                    transformed_nrj = self.energy(transformed_wav)
                    reconstructed_nrj = self.energy(reconstructed_wav)

                    transformed_stft = self.stft.compute_log1p_magnitude(transformed_wav)
                    reconstructed_stft = self.stft.compute_log1p_magnitude(reconstructed_wav)

                    batch_losses = {
                        "tL1": F.l1_loss(reconstructed_wav, transformed_wav),
                        "tL2": F.mse_loss(reconstructed_wav, transformed_wav),
                        "nrj": F.mse_loss(reconstructed_nrj, transformed_nrj).detach(),
                        "msl": F.l1_loss(transformed_stft, reconstructed_stft),
                        "mrstft": self.mrstft(reconstructed_wav.float(), transformed_wav.float()),
                    }
                    # extra model-specific terms (e.g. the adapter match loss), if any
                    batch_losses.update(extra_losses)
                    for k, value in batch_losses.items():
                        if isinstance(value, torch.Tensor) and torch.isfinite(value):
                            local_losses[k] = local_losses.get(k, 0) + value

                #aggregate local losses into global train loss
                train_loss = 0
                for _, loss in local_losses.items():
                    train_loss += loss / len(transforms)

            #skip the backprop step if the loss is ill-defined
            if not isinstance(train_loss, torch.Tensor) or not torch.isfinite(train_loss):
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
               self.save_checkpoint(stats={}, current_epoch=current_epoch, n_steps=self.n_steps, name="latest")

            #increment the step tracker
            self.n_steps += 1



    def train(self):
        """train the model on the entire dataset"""
        checkpoint = self.load_checkpoint()
        start_epoch = checkpoint.get("epoch", 0)
        self.n_steps = checkpoint.get("n_steps", 0)

        criterion = self.config.model_selection_criterion
        best_stats = self.evaluate(start_epoch)

        for current_epoch in range(start_epoch, self.config.n_epoch):
            self.train_one_epoch(current_epoch)
            curr_stats = self.evaluate(current_epoch)
            if best_stats[criterion] > curr_stats[criterion]:
                print(f"New Best model found at epoch {current_epoch}")
                best_stats = curr_stats
                self.save_checkpoint(stats={}, current_epoch=current_epoch, n_steps=self.n_steps)
            self.scheduler.step()
        self.writer.close()




if __name__ =="__main__":
    from datasets.configs import BaseDatasetConfig
    from datasets.dataset import ChunkDataset
    from sincnet.model import SincNet
    from torchinfo import summary

    learning_rate = 1e-4
    args = TrainConfig(**{
        "batch_size": 8//2,
        "n_epoch": 500,
        "sample_rate": 16_000,
        'training_id': "gtzan",
        "component": 'complex',
        "model_selection_criterion": "log-L1",
        "causal": False,
        'frame_rate': 128,
        "n_bins": 128,
        'spectrogram_scale': "mel",
        "learning_rate": learning_rate,
        "weight_decay": learning_rate / 10,
        "training_id": "gtzan",
    })

    dataset_config = BaseDatasetConfig(
        id=args.training_id,
        sample_rate=args.sample_rate
    )

    # plain SincNet autoencoder for the requested scale (trained from scratch)
    model = SincNet(
        fs=args.sample_rate,
        fps=args.frame_rate,
        scale=args.spectrogram_scale,
        component=args.component,
        n_bins=args.n_bins,
        causal=args.causal,
        decoder_type="learnt",     # the analytical decoders have no weights to train
    )
    summary(model)

    datasets = {
        split:ChunkDataset(
            parquet_file_path=dataset_config.parquet,
            h5_file_path = dataset_config.hdf5,
            split=split,
        )
        for split in ["train", "test"]
    }

    trainer = SincNetTrainer(
        model=model,
        train_set=datasets["train"],
        val_set=datasets["test"],
        config=args
    )

    trainer.train()
