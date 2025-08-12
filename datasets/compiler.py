import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
from collections import deque

import dotenv
dotenv.load_dotenv()

from datasets.utils.hdf5writer import H5Writer
from datasets.utils.waveform import WaveformLoader



class ChunkDatasetIterator:
    def __init__(self, dataset_path:str, chunk_length:int, hop_length:int, sample_rate:int):
        self.dataset_path = Path(dataset_path)
        self.name = self.dataset_path.parent.name
        self.split = self.dataset_path.name
        self.chunk_length = chunk_length  # seconds
        self.hop_length = hop_length      # seconds
        self.sr = sample_rate
        self.loader = WaveformLoader(
            sample_rate=sample_rate, 
            chunk_duration=self.chunk_length
        ) 
        self.tracks = [p for p in self.dataset_path.iterdir() if p.is_file()]


    def __len__(self):
        return len(self.tracks)

    def __iter__(self):
        chunk_samples = int(self.chunk_length * self.sr)
        hop_samples = int(self.hop_length * self.sr)
        n_chunks_required = int(np.ceil(chunk_samples / hop_samples))

        for track in tqdm(self.tracks, ncols=80, total=len(self.tracks), leave=False):
            # one hop-sized chunk stream per stem
            stream = self.loader.stream(
                query=track.as_posix(),
                offset=0.0,
                chunk_duration=self.hop_length,
            )

            # initialise buffers per stem (primed to first full window)
            buffer = deque(maxlen=n_chunks_required)
            
            while True:
                # reach the chunks
                try:
                    chunk = np.asarray(next(stream), dtype=np.float32)
                    if len(chunk) < hop_samples:
                        raise StopIteration
                    buffer.append(chunk)
                except StopIteration:
                    break
        
                # skip if the buffers are to small
                if not (len(buffer) == n_chunks_required):
                    continue

                # generate a train or test sample
                wav = np.concatenate(buffer)
                wav = wav[:chunk_samples]
                wav = self.loader.transform(wav, target_duration=self.chunk_length)
                yield {"audio": wav}



class MultiDatasetIterator:
    def __init__(self, datasets_config:list[dict], max_samples_per_dataset:int=None):
        self.max_samples_per_dataset = max_samples_per_dataset
        self.datasets = []
        for config in datasets_config:
            self.datasets.append(ChunkDatasetIterator(**config))
    
    def __iter__(self):
        progress_bar = tqdm(self.datasets, ncols=100, disable=False)
        for dataset in progress_bar:
            name = dataset.name
            split = dataset.split
            progress_bar.set_description(f'{name} | {split}')

            index = 0
            sample_progress_bar = tqdm(dataset, ncols=100, disable=True, leave=False)
            for batch in sample_progress_bar:
                batch['dataset'] = name
                batch['split'] = split
                yield batch
                index += 1

                #print(index, self.max_samples_per_dataset)
                if self.max_samples_per_dataset and index >= self.max_samples_per_dataset:
                    break


class Compiler:
    def __init__(self, compilation_file:str, iterator:MultiDatasetIterator, writer:H5Writer):
        self.iterator = iterator
        self.writer = writer
        self.compilation_file = compilation_file

    def run(self, max_samples:int=None):
        print("[Compilation][full data] starting...")
        rows = []
        h5_index = 0

        for batch in self.iterator:
            self.writer.write_batch({
                "audio": batch["audio"],
            })
            rows.append({
                "h5_index": h5_index,
                "split": batch["split"],
                "dataset": batch["dataset"],
            })

            h5_index += 1
            if max_samples and h5_index >= max_samples:
                break

        pd.DataFrame(rows).to_parquet(self.compilation_file)
        print("[Compilation] completed...")





if __name__ == "__main__":
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
    CHUNK_DURATION = int(os.getenv("CHUNK_DURATION"))
    HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
    N_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
    ROOT = os.getenv("ROOT_DIRECTORY")

    buffer_schemas = {
        "audio": (1, N_SAMPLES), 
    }

    datasets_folders = [
        #'gtzan_22050/train',
        'djmax_respectv_22050/train',
    ]

    datasets_config = [
        {
            'dataset_path':  os.path.join(ROOT, folder), 
            'chunk_length': CHUNK_DURATION, 
            "hop_length": HOP_LENGTH, 
            "sample_rate": SAMPLE_RATE,
        }
        for folder in datasets_folders
    ]

    compile_name = "gtzan"
    compile_dir = os.path.join(ROOT, "_compiled")
    os.makedirs(compile_dir, exist_ok=True)

    h5_file = os.path.join(compile_dir, f"{compile_name}_fs{SAMPLE_RATE}.hdf5")
    compilation_file = os.path.join(compile_dir, f"{compile_name}_fs{SAMPLE_RATE}.parquet")

    max_samples_per_dataset = None
    iterator = MultiDatasetIterator(datasets_config, max_samples_per_dataset=max_samples_per_dataset)

    writer = H5Writer(file_path=h5_file, buffer_schemas=buffer_schemas)
    try:
        compiler = Compiler(compilation_file=compilation_file, iterator=iterator, writer=writer)
        compiler.run()
    finally:
        writer.close()

