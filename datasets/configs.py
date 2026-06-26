import os
import dotenv
dotenv.load_dotenv()



current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)


class BaseDatasetConfig:
    id :str
    sample_rate:int 
    audio_root: str = os.getenv("H5_DIRECTORY")

    def __init__(self, id:str, sample_rate:int=None) -> None:
        super().__init__()
        self.sample_rate = sample_rate or int(os.getenv("SAMPLE_RATE", 16000))
        self.id = id

    @property
    def parquet(self) -> str:
        """Return the path to the parquet file with the dataset spec"""
        return os.path.join(self.audio_root, f"{self.id}_fs{self.sample_rate}.parquet")
    
    @property
    def hdf5(self) -> str:
        """Return the path to the model configuration"""
        return os.path.join(self.audio_root, f"{self.id}_fs{self.sample_rate}.hdf5")


class GTZANConfig(BaseDatasetConfig):
    """ traceability: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification """
    def __init__(self, sample_rate:int=None) -> None:
        super().__init__(id="gtzan", sample_rate=sample_rate)