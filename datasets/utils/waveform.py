import numpy as np
#import soundfile as sf
import pyloudnorm as pyln
import miniaudio
import subprocess
import tempfile
from datetime import timedelta

import warnings
from typing import Generator

warnings.filterwarnings("ignore")


#NOTE: for some reasons it seems that miniaudio has a faster read speed that ffmpeg
#we should check if it is linked to the platform (OSX, Linus, Windows) or there are
#some internal optimisation which is done


def stream_ffmpeg(query:str|bytes, sample_rate:int, duration:int, offset:float=0, channels:int=1):
    """ get a ffmpeg streamer to buffer a large audio file """
    file = None
    if isinstance(query, bytes):
        file = tempfile.NamedTemporaryFile("w+b", prefix="pytemp_audio_", suffix=".query")
        file.write(query)
        file.seek(0)
        query = file.name

    command = [
        "ffmpeg",
        "-v", "fatal",
        "-hide_banner",
        "-nostdin",
        "-ss", str(timedelta(seconds=offset)),
        "-vn",
        "-i", query,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-"]

    sample_width = 2 # 16 bit pcm
    max_int16 = np.iinfo(np.int16).max
    frames_to_read = int(sample_width * sample_rate * duration)
    process = subprocess.Popen(command, stdin=None, stdout=subprocess.PIPE, bufsize=10**8)
    while True:
        # Read decoded audio frames from stdout process.
        buffer = process.stdout.read(frames_to_read)
        # Break the loop if buffer length is not matching target.
        if len(buffer)!= frames_to_read:
            break
        waveform = np.frombuffer(buffer, np.int16)
        yield waveform / max_int16

    process.terminate()
    if file: file.close()



def stream_miniaudio(query:str| bytes, sample_rate:int, duration:int, offset:int=0, channels:int=1) -> Generator:
    """ get a miniaudio streamer to buffer a large audio file """
    key = "data" if isinstance(query, bytes) else "filename"
    params = {
        **{key:query},
        "sample_rate": sample_rate,
        "frames_to_read": int(duration * sample_rate),
        "output_format": miniaudio.SampleFormat.FLOAT32,
        "seek_frame": int(offset * sample_rate),
        "nchannels":channels
    }

    if isinstance(query, str):
        params["seek_frame"] = int(offset * sample_rate)

    stream = miniaudio.stream_memory(**params) if key=="data" else miniaudio.stream_file(**params)
    return stream


class WaveformLoader:
    """
    A modular loader for the offline dataset creation pipeline.
    It uses librosa to handle loading and resampling in a single step and
    consistently uses the (channels, samples) layout for PyTorch compatibility.
    """
    def __init__(self, sample_rate: int):
        print(f"Initializing PreprocessingWaveformLoader with sample rate sr:{sample_rate}...")
        self.sr = sample_rate
        try:
            self.meter = pyln.Meter(self.sr)
        except Exception as e:
            print(f"Failed to create loudness meter: {e}")
            self.meter = None


    def load_audio(self, audio_path: str, nchannels:int=2) -> np.ndarray:
        """ Loads and resamples an audio file using librosa, returning a (channels, samples) array """
        try:
            # Step 1: Decode the file to get the object with info and the sample buffer.
            decoded_file = miniaudio.decode_file(
                audio_path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                sample_rate=self.sr,
                nchannels=nchannels
            )

            # Step 2: Access the raw sample buffer from the .samples attribute.
            sample_buffer = decoded_file.samples

            # Step 3: Use np.frombuffer to interpret the raw bytes as float32 numbers.
            # This will create a 1D (flat) array.
            waveform_flat = np.frombuffer(sample_buffer, dtype=np.float32)
            
            # Step 4: Reshape the flat array into (samples, channels) using the file's metadata.
            num_channels = decoded_file.nchannels
            num_frames = decoded_file.num_frames
            waveform_shaped = waveform_flat.reshape(num_frames, num_channels)

            # Step 5: Transpose to get the desired (channels, samples) layout
            if waveform_shaped.ndim > 1:
                waveform = waveform_shaped.T
            else:
                waveform = np.expand_dims(waveform_shaped, axis=0)
                
            return waveform
            
        except Exception as e:
            print(f"Error loading audio from {audio_path} with miniaudio: {e}")
            return None
    

    def stream(self, query:str|bytes, offset:float=0, chunk_duration:float=5, nchannels:int=2) -> Generator:
        """ read chunks from a large audio file """
        params = {"query":query, "sample_rate":self.sr, "duration":chunk_duration, "offset":offset, "channels":nchannels}
        if isinstance(query, bytes) or query.endswith((".m4a", ".aac")):
            return stream_ffmpeg(**params)
        return stream_miniaudio(**params)
    

    def load_segment(self, audio_path: str, duration: float,  offset: float=0, nchannels:int=2) -> np.ndarray:
        """
        Loads a specific segment of an audio file from a given offset and for a given duration.

        Args:
            audio_path (str): Path to the audio file.
            offset (float): Start time of the segment in seconds.
            duration (float): Duration of the segment to load in seconds.

        Returns:
            np.ndarray: The requested audio segment as a (channels, samples) array, or None on failure.
        """
        try:
            stream = self.stream(query=audio_path, offset=offset, chunk_duration=duration, nchannels=nchannels)
            waveform_flat = next(stream)
            waveform_flat = np.asarray(waveform_flat)
            del stream

            # Step 3: Convert the buffer to a NumPy array, same as in load_audio.
            #waveform_flat = np.frombuffer(decoded_file.samples, dtype=np.float32)
            waveform_shaped = waveform_flat.reshape(-1, nchannels)

            if waveform_shaped.ndim > 1:
                waveform = waveform_shaped.T
            else:
                waveform = np.expand_dims(waveform_shaped, axis=0)

            # Step 4: Slice the loaded waveform to the desired duration.
            return waveform
        except Exception as e:
            print(f"Error loading segment from {audio_path}: {e}")
            return None
        

    def normalise_loudness(self, audio: np.ndarray, original_lufs: float, target_lufs: float) -> np.ndarray:
        # This function is shape-agnostic and requires no changes.
        normalized_audio = pyln.normalize.loudness(audio, original_lufs, target_lufs)
        peak = np.max(np.abs(normalized_audio))
        if peak > 1.0:
            normalized_audio = normalized_audio / peak
        return normalized_audio


    def measure_loudness(self, audio: np.ndarray) -> float:
        """Measures the integrated loudness of a (channels, samples) waveform."""
        if self.meter is None:
            raise RuntimeError("Loudness meter was not initialized.")
        
        # CHANGED: For a (channels, samples) array, we average across axis=0 to get mono.
        if audio.shape[0] > 1:
            # Average the channels to create a mono signal for measurement
            audio_for_measure = np.mean(audio, axis=0)
        else:
            audio_for_measure = audio.flatten()
        return self.meter.integrated_loudness(audio_for_measure)


    def get_chunks(self, audio: np.ndarray, chunk_duration: float, hop_duration: float) -> Generator[np.ndarray, None, None]:
        """ A generator that yields chunks from a (channels, samples) waveform. """
        chunk_samples = int(chunk_duration * self.sr)
        hop_samples = int(hop_duration * self.sr)        
        num_samples = audio.shape[-1]

        for start_sample in range(0, num_samples - chunk_samples + 1, hop_samples):
            end_sample = start_sample + chunk_samples
            yield audio[..., start_sample:end_sample]



if __name__ == '__main__':

    audio_path = "audio/space-of-soul.mp3"
    loader = WaveformLoader(sample_rate=16000)
    #audio = loader.load_audio(audio_path)
    audio = loader.load_segment(audio_path, 5, 5)

    for x in loader.get_chunks(audio, chunk_duration=5, hop_duration=5):
        print(x.shape)
 