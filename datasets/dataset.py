import torch
import pandas as pd
import h5pickle as h5py
from torch.utils.data import Dataset



class ChunkDataset(Dataset):

    def __init__(self, parquet_file_path:str, h5_file_path:str, split:str=None, **kwargs):
        self.h5_file_path = h5_file_path
        self.init_table_readers(h5_file_path)
        self.df = pd.read_parquet(parquet_file_path)
        if split :
            print(f"Loading dataset: {split} from {parquet_file_path}")
            self.df = self.df[self.df["split"] == split]
        else:
            print(f"Loading dataset: <ALL> from {parquet_file_path}")
        self.df.sort_values(by=["h5_index"], inplace=True)
        print(self.df["split"].value_counts())

    def init_table_readers(self, h5_file_path:str):
        print(f"Reading from {h5_file_path}")
        table = h5py.File(h5_file_path, mode='r')
        self.audio = table["audio"]

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index:int):
        row = self.df.iloc[index]
        h5_index = row.h5_index
        audio = self.audio[h5_index]
        samples = {"h5_index":h5_index}
        samples["waveform"] = torch.from_numpy(audio).float()
        return samples



if __name__ =="__main__":
    from torch.utils.data import DataLoader
    from datasets.configs import BaseDatasetConfig
    
    dataset_config = BaseDatasetConfig(id="gtzan")

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