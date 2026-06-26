import json
import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import numpy as np


class StubWaveformLoader:
    def __init__(self, *args, **kwargs):
        pass


def stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def import_compiler_with_optional_stubs():
    stubs = {
        "tqdm": stub_module("tqdm", tqdm=lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []),
        "pandas": stub_module("pandas", DataFrame=None),
        "compress_json": stub_module("compress_json", dump=lambda data, path: None),
        "dotenv": stub_module("dotenv", load_dotenv=lambda: None),
        "datasets.utils.hdf5writer": stub_module("datasets.utils.hdf5writer", H5Writer=object),
        "datasets.utils.waveform": stub_module("datasets.utils.waveform", WaveformLoader=StubWaveformLoader),
    }
    originals = {name: sys.modules.get(name) for name in stubs}
    try:
        for name, module in stubs.items():
            sys.modules.setdefault(name, module)
        return importlib.import_module("datasets.compiler")
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


compiler_module = import_compiler_with_optional_stubs()
ChunkDatasetIterator = compiler_module.ChunkDatasetIterator
Compiler = compiler_module.Compiler
MultiDatasetIterator = compiler_module.MultiDatasetIterator


class FakeWaveformLoader:
    def __init__(self, audio: np.ndarray):
        self.audio = audio

    def load_audio(self, audio_path: str, nchannels: int) -> np.ndarray:
        return self.audio

    def measure_loudness(self, audio: np.ndarray) -> float:
        return -23.0

    def normalise_loudness(self, audio: np.ndarray, original_lufs: float, target_lufs: float) -> np.ndarray:
        return audio

    def get_chunks(self, audio: np.ndarray, chunk_duration: float, hop_duration: float):
        chunk_samples = int(chunk_duration)
        hop_samples = int(hop_duration)
        for start in range(0, audio.shape[-1] - chunk_samples + 1, hop_samples):
            yield audio[..., start : start + chunk_samples]


def test_chunk_iterator_concatenates_overlap_buffer_on_time_axis(tmp_path):
    dataset_path = tmp_path / "gtzan" / "train"
    dataset_path.mkdir(parents=True)
    (dataset_path / "track.wav").touch()

    audio = np.arange(10, dtype=np.float32).reshape(1, -1)
    iterator = ChunkDatasetIterator(
        dataset_path=dataset_path.as_posix(),
        target_loudness_lufs=-23.0,
        n_channels=1,
        chunk_length=4,
        hop_length=2,
        sample_rate=1,
    )
    iterator.loader = FakeWaveformLoader(audio)

    batches = list(iterator)

    assert batches[0]["audio"].shape == (1, 4)
    np.testing.assert_array_equal(batches[0]["audio"], np.array([[0, 1, 2, 3]], dtype=np.float32))
    np.testing.assert_array_equal(batches[1]["audio"], np.array([[2, 3, 4, 5]], dtype=np.float32))


def test_peak_stats_are_json_serializable_python_floats():
    iterator = MultiDatasetIterator.__new__(MultiDatasetIterator)
    iterator.datasets = [
        SimpleNamespace(peaks=[np.float32(0.5), np.float64(0.75)]),
        SimpleNamespace(peaks=[np.float32(1.0)]),
    ]

    results = iterator.compute_peaks_stats()

    json.dumps(results)
    assert all(isinstance(peak, float) for peak in results["peaks"])
    assert all(isinstance(value, float) for value in results["stats"].values())


def test_compiler_writes_peak_stats_to_configured_path(monkeypatch, tmp_path):
    parquet_path = tmp_path / "compiled.parquet"
    peaks_path = tmp_path / "compiled.peaks.json"
    calls = {}

    class FakeDataFrame:
        def __init__(self, rows):
            self.rows = rows

        def to_parquet(self, path):
            calls["parquet_path"] = path

    class FakeIterator:
        def __iter__(self):
            yield {"audio": np.zeros((1, 4), dtype=np.float32), "dataset": "gtzan", "split": "train"}

        def compute_peaks_stats(self):
            return {"peaks": [0.5], "stats": {"mean": 0.5}}

    class FakeWriter:
        def write_batch(self, batch):
            calls["batch"] = batch

    def fake_dump(data, path):
        calls["dump"] = (data, path)

    monkeypatch.setattr(compiler_module.pd, "DataFrame", FakeDataFrame)
    monkeypatch.setattr(compiler_module.compress_json, "dump", fake_dump)

    compiler = Compiler(
        compilation_file=parquet_path.as_posix(),
        peaks_file=peaks_path.as_posix(),
        iterator=FakeIterator(),
        writer=FakeWriter(),
    )
    compiler.run()

    assert calls["parquet_path"] == parquet_path.as_posix()
    assert calls["dump"][1] == peaks_path.as_posix()
    assert calls["batch"]["audio"].shape == (1, 4)
