import numpy as np
import librosa
import miniaudio
import subprocess
import tempfile
from datetime import timedelta

import warnings
import traceback
warnings.filterwarnings("ignore")

from typing import Union

#NOTE: for some reasons it seems that miniaudio has a faster read speed that ffmpeg
#we should check if it is linked to the platform (OSX, Linus, Windows) or there are
#some internal optimisation which is done



def stream_ffmpeg(query:Union[str, bytes], sample_rate:int, duration:int, offset:float=0, channels:int=1):
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



def stream_miniaudio(query:Union[str, bytes], sample_rate:int, duration:int, offset:int=0, channels:int=1):
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
    return map(np.asarray, stream)



def adjust_length_along_axis(array:np.ndarray, target_length:int, axis:int) -> np.ndarray:
    """ pad an array along a given axis to match a target length"""
    pad_size = target_length - array.shape[axis]
    if pad_size==0:
        return array
    elif pad_size < 0:
        return array.take(indices=range(0, target_length), axis=axis)
    else:
        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)
        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)



class WaveformLoader:

    def __init__(self, sample_rate:Union[int, str], chunk_duration:Union[float, str], **kwargs):
        """ duration : duration in seconds of a chunk | sr : sampling rate """
        print(f"Initialising waveform loader with sample rate sr:{sample_rate}...")
        self.sr = int(sample_rate)
        self.chunk_duration = float(chunk_duration)
        self.chunk_length = int(sample_rate * chunk_duration)
        self.features_dimensions = self.chunk_length


    def load(self, audio_path:str, offset:float=0, duration:float=None) -> object:
        """ load a waveform and transform it to get the audio features """
        try:
            waveform, sr = librosa.load(path=audio_path, sr=self.sr, offset=offset,
                duration=duration if duration else self.chunk_duration)
            return self.transform(waveform, duration).astype(np.float32)
        except Exception as e:
            print("=====================PROBLEM WITH=============================")
            print({"audio_path":audio_path, "offset":offset, "duration":duration, "sr":self.sr, "chunk_duration":self.chunk_duration})
            print("===============================================================")
            print(traceback.format_exc())
            raise e

    def transform(self, waveform:np.ndarray, target_duration:float=None) -> np.ndarray:
        """ transform a waveform into a feature """
        target_length = int(self.sr * target_duration) if target_duration else self.chunk_length
        waveform = adjust_length_along_axis(waveform, target_length=target_length, axis=0)

        #normalisation from [wmin, wmax] to [-1, 1]
        wmin = waveform.min()
        wmax = waveform.max()
        if wmin != wmax:
            waveform = 2 * (waveform - wmin) / (wmax - wmin) - 1
        return np.expand_dims(waveform, axis=0)


    def stream(self, query:Union[str, bytes], offset:float=None, chunk_duration:float=None):
        """ read chunks from a large audio file """
        chunk_duration = chunk_duration if chunk_duration else self.chunk_duration
        offset = offset if offset else 0
        params = {"query":query, "sample_rate":self.sr, "duration":chunk_duration, "offset":offset}

        if isinstance(query, bytes) or query.endswith((".m4a", ".aac")):
            return stream_ffmpeg(**params)
        return stream_miniaudio(**params)


if __name__ == '__main__':

    audio_path = "audio/space-of-soul.mp3"
    loader = WaveformLoader(**{"sample_rate":16000, "chunk_duration":5.0})
    data = loader.stream(audio_path)
    for x in data:
        x = loader.transform(x)
        print(x.shape)
 