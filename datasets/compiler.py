import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
from collections import deque
import compress_json

import dotenv
dotenv.load_dotenv()

from datasets.utils.hdf5writer import H5Writer
from datasets.utils.waveform import WaveformLoader



class ChunkDatasetIterator:
    def __init__(self, dataset_path:str, target_loudness_lufs:float, n_channels:int, chunk_length:int, hop_length:int, sample_rate:int):
        self.dataset_path = Path(dataset_path)
        self.name = self.dataset_path.parent.name
        self.split = self.dataset_path.name
        self.target_loudness_lufs = target_loudness_lufs
        self.n_channels = n_channels #1=mono, 2=stereo
        self.chunk_length = chunk_length  # seconds
        self.hop_length = hop_length      # seconds
        self.sr = sample_rate
        self.loader = WaveformLoader(sample_rate=sample_rate) 
        self.tracks = [p for p in self.dataset_path.iterdir() if p.is_file()]
        self.peaks = []

    def __len__(self):
        return len(self.tracks)

    def __iter__(self):
        chunk_samples = int(self.chunk_length * self.sr)
        hop_samples = int(self.hop_length * self.sr)
        n_chunks_required = int(np.ceil(chunk_samples / hop_samples))

        for track in tqdm(self.tracks, ncols=80, total=len(self.tracks), leave=False):
            # one hop-sized chunk stream per stem
            audio = self.loader.load_audio(
                audio_path=track.as_posix(),
                nchannels=self.n_channels
            )

            #skip pathological cases
            if audio is None:
                continue
            
            loudness = self.loader.measure_loudness(audio)
            audio = self.loader.normalise_loudness(
                audio, 
                original_lufs=loudness, 
                target_lufs=self.target_loudness_lufs
            )
            self.peaks.append(np.max(np.abs(audio)))
            stream = self.loader.get_chunks(
                audio=audio, 
                chunk_duration=self.chunk_length, 
                hop_duration=self.hop_length
            )
            # initialise buffers per stem (primed to first full window)
            buffer = deque(maxlen=n_chunks_required)
            
            while True:
                # reach the chunks
                try:
                    chunk = next(stream)
                    if chunk.shape[-1] < hop_samples:
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
                yield {"audio": wav}



class MultiDatasetIterator:
    def __init__(self, datasets_config:list[dict], max_samples_per_dataset:int=None):
        self.max_samples_per_dataset = max_samples_per_dataset
        self.datasets : list[ChunkDatasetIterator] = []
        for config in datasets_config:
            self.datasets.append(ChunkDatasetIterator(**config))

    def compute_peaks_stats(self) -> dict:
        peaks = []
        for dataset in self.datasets:
            peaks.extend(dataset.peaks)
        stats = {
            'mean': np.mean(peaks),
            'std': np.std(peaks),
            'p95': np.percentile(peaks, 95),
            'p99': np.percentile(peaks, 99),
            'max': np.max(peaks)
        }
        print(f"Peak statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  95th percentile: {stats['p95']:.3f}")
        print(f"  99th percentile: {stats['p99']:.3f}")
        print(f"  Max observed: {stats['max']:.3f}")
        results = {
            "peaks": peaks,
            "stats": stats
        }
        return results
    
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
    def __init__(self, compilation_file:str, peaks_file:str, iterator:MultiDatasetIterator, writer:H5Writer):
        self.iterator = iterator
        self.writer = writer
        self.compilation_file = compilation_file
        self.peaks_file = peaks_file

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
        compress_json.dump(self.iterator.compute_peaks_stats(), peaks_file)
        print("[Compilation] completed...")





if __name__ == "__main__":
    ROOT = os.getenv("ROOT_DIRECTORY")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
    CHUNK_DURATION = int(os.getenv("CHUNK_DURATION"))
    HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
    N_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
    TARGET_LUFFS = -float(os.getenv("TARGET_LUFFS", 23))
    N_CHANNELS = 1

    buffer_schemas = {
        "audio": (N_CHANNELS, N_SAMPLES), 
    }

    datasets_folders = [
        'gtzan/train',
    ]

    datasets_config = [
        {
            'dataset_path': os.path.join(ROOT, folder), 
            'chunk_length': CHUNK_DURATION, 
            "hop_length": HOP_LENGTH, 
            "sample_rate": SAMPLE_RATE,
            "n_channels": N_CHANNELS,
            "target_loudness_lufs": TARGET_LUFFS
        }
        for folder in datasets_folders
    ]

    compile_name = "gtzan"
    compile_dir = os.path.join(ROOT, "_compiled")
    os.makedirs(compile_dir, exist_ok=True)

    h5_file = os.path.join(compile_dir, f"{compile_name}_fs{SAMPLE_RATE}.hdf5")
    compilation_file = os.path.join(compile_dir, f"{compile_name}_fs{SAMPLE_RATE}.parquet")
    peaks_file = os.path.join(compile_dir, f"{compile_name}_fs{SAMPLE_RATE}.peaks.json")

    max_samples_per_dataset = None
    iterator = MultiDatasetIterator(datasets_config, max_samples_per_dataset=max_samples_per_dataset)

    writer = H5Writer(file_path=h5_file, buffer_schemas=buffer_schemas)
    try:
        compiler = Compiler(compilation_file=compilation_file, peaks_file=peaks_file, iterator=iterator, writer=writer)
        compiler.run()
    finally:
        writer.close()

