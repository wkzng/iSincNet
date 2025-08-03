import os
import dotenv
dotenv.load_dotenv()

H5_DIRECTORY = os.getenv("H5_DIRECTORY")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)


class BaseDatasetConfig:
    id :str
    sample_rate:int = SAMPLE_RATE
    metadata_root:str = os.path.join(project_root, "datasets", "compiled")
    audio_root: str = H5_DIRECTORY

    @property
    def parquet(self) -> str:
        """Return the path to the parquet file with the dataset spec"""
        return os.path.join(self.metadata_root, self.id, "dataset.parquet")
    
    @property
    def config(self) -> str:
        """Return the path to the model configuration"""
        return os.path.join(self.metadata_root, self.id, "model_config.json")
    
    @property
    def hdf5(self) -> str:
        """Return the path to the model configuration"""
        return os.path.join(self.audio_root, f"{self.id}_fs{self.sample_rate}.h5")


class GTZANConfig(BaseDatasetConfig):
    id :str="gtzan"


class EDMConfig(BaseDatasetConfig):
    id :str="edm"