

import numpy as np
import tables
import queue
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.dataset import ChunkDataset



class H5Writer:

    def __init__(self, file_path:str, buffer_params:dict, mode:str):
        self.h5, self.buffers = self.initialize_buffer(file_path, buffer_params, mode)
        self.q = queue.Queue()
        self.run = True
        
    def initialize_buffer(self, file_path:str, buffer_params:dict, mode:str):
        if mode=="w" and os.path.exists(file_path):
            os.remove(file_path)
        if mode=="a" and not os.path.exists(file_path):
            mode = "w"
        h5 = tables.open_file(file_path, mode)
        buffers = []
        for name, dimensions in buffer_params.items():
            if isinstance(dimensions, tuple):
                shape = (0, *dimensions)
            elif isinstance(dimensions, int):
                shape = (0, dimensions)
            buffers.append(h5.create_earray(h5.root, name, tables.Float16Atom(), shape))
        return h5, buffers

    def start(self):
        while self.run:
            for idx, datum in enumerate(self.q.get()):
                self.buffers[idx].append(np.expand_dims(datum, axis=0))
            self.q.task_done()

    def write(self, data):
        for idx, datum in enumerate(data):
            self.buffers[idx].append(np.expand_dims(datum, axis=0))

    def close(self):
        self.h5.close()



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets.configs import GTZANConfig
    
    dataset_config = GTZANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChunkDataset(
        parquet_file_path=dataset_config.parquet,
        h5_file_path = dataset_config.hdf5,
        load_from_hdf5=False,
        mode="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64, 
        num_workers=8, 
        pin_memory=False,
        shuffle=False
    )

    sample_rate = dataset_config.sample_rate
    duration = dataset.duration
    n_samples = int(sample_rate * duration)
    n_batches = len(dataloader)
    h5_index = 0

    try:
        h5_writer = H5Writer(
            file_path = dataset_config.hdf5,
            buffer_params = {"waveform":n_samples},
            mode="w"
        )

        progress_bar = tqdm(enumerate(dataloader), ncols=100, total=n_batches, disable=False)
        with torch.no_grad():
            for _, batch in progress_bar:
                waveforms = batch["waveform"]
                samples_indices = batch["h5_index"]
                batch_size = samples_indices.shape[0]

                for i in range(batch_size):
                    waveform = waveforms[i]
                    sample_h5_index = int(samples_indices[i])
                    assert sample_h5_index == h5_index
                    h5_writer.write(waveform)
                    h5_index +=1
    finally:
        h5_writer.close()
