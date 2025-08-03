import torch
import pandas as pd
import numpy as np
import h5pickle as h5py
from torch.utils.data import Dataset
from datasets.utils.waveform import WaveformLoader



class ChunkDataset(Dataset):
    """ Dataset for audio embedding"""

    def __init__(self, parquet_file_path:str,  h5_file_path:str, mode:str=None, load_from_hdf5:bool=False, **kwargs):
        self.df = pd.read_parquet(parquet_file_path)
        self.duration = int(self.df.iloc[0].duration)
        self.sample_rate = int(self.df.iloc[0].sample_rate)
        self.audio_loader = WaveformLoader(sample_rate=self.sample_rate, chunk_duration=self.duration)
        self.load_from_hdf5 = load_from_hdf5

        if mode :
            print(f"Loading dataset: {mode} from {parquet_file_path}")
            self.df = self.df[self.df["mode"] == mode]

        if load_from_hdf5:
            self.init_table_readers(h5_file_path)
        print(f"Dataset size: {len(self.df)}\n")


    def init_table_readers(self, h5_file_path:str):
        print(f"Reading from {h5_file_path}")
        self.table = h5py.File(h5_file_path, mode='r')
        self.size = self.table["waveform"].shape[0]


    def __len__(self) -> int:
        return len(self.df)
    

    def load_from_audio_file(self, row:dict | pd.Series) -> np.ndarray:
        endpoints = row.endpoints
        if isinstance(endpoints[0], np.int64):
            start = row.endpoints[0]
        else:
            endpoints = np.random.choice(row.endpoints, size=1)[0]
            start = endpoints[0]

        return self.audio_loader.load(
            audio_path=row.file_path, 
            offset=start, 
            duration=row.duration
        )

    def __getitem__(self, index:int):
        row = self.df.iloc[index]
        h5_index = row.h5_index

        if self.load_from_hdf5:
            waveform = self.table["waveform"][h5_index]
        else:
            waveform = self.load_from_audio_file(row)

        samples = {"h5_index":h5_index}
        samples["waveform"] = torch.from_numpy(waveform).float()
        return samples



if __name__ =="__main__":
    from torch.utils.data import DataLoader
    from datasets.configs import GTZANConfig
    
    dataset_config = GTZANConfig()

    ds = ChunkDataset(
        parquet_file_path=dataset_config.parquet,
        h5_file_path = dataset_config.hdf5,
        mode="train",
    )
    
    data_loader = DataLoader(dataset=ds, batch_size=10)
    for batch in data_loader:
        for k, v in batch.items():
            print("=================")
            print(k, v.shape)
            print(v.min(), v.max())
        exit()