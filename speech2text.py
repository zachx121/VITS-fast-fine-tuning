import whisper
import sys
import numpy as np
import ffmpeg
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as write_wav
import logging

logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class Speech2Text:
    def __init__(self, model_type, download_root="./whisper_models"):
        self.model_type = model_type
        self.download_root = download_root
        self.model = None
        self.is_init = False

    def init(self, prehot_audio="./prehot_speech2text.wav"):
        logging.info(">>> loading whiser model")
        self.model = whisper.load_model(self.model_type, download_root=self.download_root)
        logging.info(">>> loading whiser model(done.)")
        try:
            logging.info(">>> try to transcribe with prehot_audio")
            self.transcribe(prehot_audio, fp16=False)
        except (Exception,) as e:
            logging.warning("pre_hot fail.")
        self.is_init = True
        return self

    def transcribe(self, audio_file, **kwargs):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        result = self.model.transcribe(audio_file,
                                       task="transcribe",
                                       beam_size=5,
                                       best_of=5,
                                       word_timestamps=False,
                                       **kwargs)
        return result['text']

    def transcribe_buffer(self, audio_buffer, sr_inp, channels_inp, **kwargs):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        audio = self.load_audio_raw(audio_buffer, sr_inp=sr_inp, channels=channels_inp)
        result = self.model.transcribe(audio,
                                       task="transcribe",
                                       beam_size=5,
                                       best_of=5,
                                       word_timestamps=False,
                                       **kwargs)
        return result['text']

    @staticmethod
    def load_audio_raw(pcm_data: bytes,
                       sr_inp: int = 44100,
                       channels: int = 2,
                       sample_fmt: str = 's16le',
                       sr_out: int = 16000):
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input('pipe:', format=sample_fmt, acodec='pcm_s16le', ac=channels, ar=sr_inp)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr_out)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=pcm_data)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: (str, bytes), sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


load_audio_raw = Speech2Text.load_audio_raw


if __name__ == '__main__':
    # process file.
    if False:
        MODEL_TYPE = sys.argv[1] if len(sys.argv) >= 2 else "tiny"
        logging.info(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
        M_stt = Speech2Text(model_type=MODEL_TYPE, download_root="./whisper_models").init()
        text = M_stt.transcribe("./audio_daniel_2021-part0.wav")
        logging.info(">>> transcribe_file:\n%s" % text)

    if True:
        MODEL_TYPE = "tiny"
        logging.info(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
        M_stt = Speech2Text(model_type=MODEL_TYPE, download_root="./whisper_models").init()
        rate, raw_audio = wav_read('/Users/didi/0-Code/samples/CXM/audio_daniel_2021-part0.wav')
        if raw_audio.dtype == np.int16:
            sampwidth = 2
        elif raw_audio.dtype == np.int32:
            sampwidth = 4
        else:
            raise Exception("unexpected type: %s" % raw_audio.dtype)
        slice_length = 3 * rate
        for i in range(0, len(raw_audio), slice_length):
            audio_slice = raw_audio[i:i + slice_length]
            data = {"audio": audio_slice.tobytes(),
                    "audio_shape": audio_slice.shape,
                    "sample_rate": rate}
            text = M_stt.transcribe_buffer(data['audio'],
                                           sr_inp=data['sample_rate'],
                                           channels_inp=data['audio_shape'][1],
                                           fp16=False)
            print(text)

        sys.exit(0)
        audio_slice = raw_audio[0:0 + slice_length]
        data = {"audio": audio_slice.tobytes(), "audio_shape": audio_slice.shape, "sample_rate": rate}
        # MOCK
        audio_buffer = data['audio']
        audio_shape = data['audio_shape']
        audio_sr = data['sample_rate']
        audio_raw = np.frombuffer(audio_buffer, np.int16).flatten().astype(np.float32) * (1.0 / 2 ** 15)

        # audio_slice: (132300, 2) int16, scipy.io.wavfile直接读出来的数组
        # audio_buffer: b'\x00....' bytes, 数组直接audio_slice.tobytes()的结果
        # audio_raw: (264600,) float32, 字节流转回数组并通过除以2^15次归一化
        #   - audio_raw.reshape(audio_shape)*2**15 就变回 audio_slice了
        #
        # load_audio("./tmp2.wav"): (48000,) float32,
        #   - 文件由audio_raw使用scipy.io.wavfile.write，reshape(-1,2)后写入
        #   - 原始数组audio_raw的采样率是 44100
        #   - 函数load_audio默认采样率是 16000
        #   - e.g. 44100/16000==audio_raw.reshape(-1,2).shape[0]/load_audio("./tmp2.wav").shape[0]
        # load_audio(audio_buffer): (154345,) float32,

        # get correct result | re-construct & write to file.
        write_wav("./tmp2.wav", audio_sr, audio_raw.reshape(audio_shape))
        print("from .wav: " + M_stt.transcribe("./tmp2.wav", fp16=False))
        # get correct result | load_audio with the file.
        print("from load_audio(.wav): " + M_stt.transcribe(load_audio("./tmp2.wav"), fp16=False))
        # get empty string
        print(load_audio("./tmp2.wav").shape)
        print(load_audio_raw(audio_buffer,sr=audio_sr,channels=2).shape)
        print("from load_audio_raw(buffer): " + M_stt.transcribe(load_audio_raw(audio_buffer, sr=audio_sr, channels=2), fp16=False))
        print("from load_audio(buffer): " + M_stt.transcribe(load_audio(audio_buffer), fp16=False))
        print("from audio_raw: " + M_stt.transcribe(audio_raw, fp16=False))



        #text = M_stt.transcribe_data(audio, audio_sr)
    # timeit
    # import timeit
    # time_res = timeit.timeit(lambda: M_stt.transcribe("./audio_daniel_2021-part0.wav"), number=3)
    # logging.info(">>> timeit: %.4f" % (time_res/3.0))

    # import pickle
    # with open("./result_%s.pkl" % MODEL_TYPE, "wb") as f:
    #     pickle.dump(result, f)

    # import json
    # print(json.dumps(result))
