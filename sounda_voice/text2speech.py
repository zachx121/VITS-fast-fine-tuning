import logging
import os
import re
import sys
sys.path.append("./")
print(sys.path)

import utils_audio
from scipy.io.wavfile import write
import numpy as np
import pyaudio
import librosa
from sounda_voice.function.encoder import inference as encoder
from sounda_voice.function.synthesizer.inference import Synthesizer
from sounda_voice.function.vocoder.hifigan import inference as gan_vocoder
from pathlib import Path

def synthesize(texts,
               mock_audio_fp,
               encoder_fp="~/models/pretrained1.pt",
               synth_fp="~/models/model_zn_v1.pt",
               vocoder_fp="~/models/g_hifigan.pt"
               ):
    # load function
    encoder.load_model(Path(encoder_fp))
    current_synt = Synthesizer(Path(synth_fp), verbose=False)
    gan_vocoder.load_model(Path(vocoder_fp))

    wav, sample_rate = librosa.load(mock_audio_fp)
    #print("sample_rate from librosa.load is %s" % sample_rate)
    # preprocess
    encoder_wav = encoder.preprocess_wav(wav, sample_rate)
    embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # Load input text
    punctuation = '！，。、,'  # punctuate and split/clean text
    processed_texts = []
    for text in texts.split("\n"):
        for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
            if processed_text:
                processed_texts.append(processed_text.strip())
    texts = processed_texts

    # synthesize and vocode
    embeds = [embed] * len(texts)
    specs = current_synt.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)
    sample_rate = Synthesizer.sample_rate
    wav, sample_rate = gan_vocoder.infer_waveform(spec)
    return wav, sample_rate

def play_audio(audio_buffer, sr, channels=1):
    assert isinstance(audio_buffer, bytes), "'audio_buffer' should be bytes. try .tobytes() for ndarray."
    print("use sr: %s" % sr)
    p = pyaudio.PyAudio()
    # 打开一个音频流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    output=True)
    # 播放音频
    stream.write(audio_buffer)
    # 结束后关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

def save_audio(audio, sr, save_fp, channels=1):
    write(save_fp, sr, audio)

SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS

class Text2Speech:
    def __init__(self, encoder_fp, synth_fp, vocoder_fp):
        self.encoder_fp = encoder_fp
        self.synth_fp = synth_fp
        self.vocoder_fp = vocoder_fp
        self.current_synt = None
        self.is_init = False

    def init(self):
        self.current_synt = Synthesizer(Path(self.synth_fp), verbose=False)
        encoder.load_model(Path(self.encoder_fp))
        gan_vocoder.load_model(Path(self.vocoder_fp))
        self.is_init = True

    def synth_file(self, texts, mock_audio_fp):
        wav, sample_rate = librosa.load(mock_audio_fp)
        # source_spec = Synthesizer.make_spectrogram(wav)
        print("sample_rate from librosa.load is %s" % sample_rate)
        # preprocess
        encoder_wav = encoder.preprocess_wav(wav, sample_rate)
        embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Load input text
        punctuation = '！，。、,'  # punctuate and split/clean text
        processed_texts = []
        #print("ori-text: %s" % texts)
        for text in texts.split("\n"):
            for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
                if processed_text:
                    processed_texts.append(processed_text.strip())
        texts = processed_texts
        #print("processed-text: %s" % texts)

        # synthesize and vocode
        embeds = [embed] * len(texts)
        specs = self.current_synt.synthesize_spectrograms(texts, embeds)
        spec = np.concatenate(specs, axis=1)
        sample_rate = Synthesizer.sample_rate
        wav, sample_rate = gan_vocoder.infer_waveform(spec)
        return wav, sample_rate

    def _notwork_synth(self, text, mock_audio_buffer, sample_rate=SAMPLE_RATE):
        # source_spec = Synthesizer.make_spectrogram(wav)
        # preprocess
        encoder_wav = encoder.preprocess_wav(mock_audio_buffer, sample_rate)
        embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Load input text
        punctuation = '！，。、,'  # punctuate and split/clean text
        processed_texts = []
        print("ori-text: %s" % text)
        for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
            if processed_text:
                processed_texts.append(processed_text.strip())
        logging.debug("processed text is: '%s'" % ",".join(processed_texts))
        texts = processed_texts
        print("processed-text: %s" % text)

        # synthesize and vocode
        embeds = [embed] * len(texts)
        specs = self.current_synt.synthesize_spectrograms(texts, embeds)
        spec = np.concatenate(specs, axis=1)
        sample_rate = Synthesizer.sample_rate
        wav, sample_rate = gan_vocoder.infer_waveform(spec)
        return wav, sample_rate


if __name__ == '__main__':
    import pypinyin
    pypinyin.load_phrases_dict({"映射测试": [["ce4"], ["shi4"], ["ying4"], ["she4"]]}, style=pypinyin.Style.TONE3)
    text1 = "这是一段测试录音，用来产生下面的录音内容。测试语音，短句测试"
    text2 = "连续语音测试特殊符号「测试」，阿拉伯数字测试1234，需要单独拼音映射吗，长句测试样例"

    text = text1
    mock_audio_fp = "/Users/didi/0-Code/VITS-fast-fine-tuning/voice_sample/female_p1_zh.wav"
    mock_audio_fp = "/Users/didi/0-Code/#samples/haichao.wav"

    if True:
        wav, sr = synthesize(texts=text,
                             mock_audio_fp=mock_audio_fp,
                             encoder_fp="./sounda_voice_models/encoder/pretrained1.pt",
                             synth_fp="./sounda_voice_models/synth/pre_145000.pt",
                             vocoder_fp="./sounda_voice_models/vocoder/g_hifigan.pt")
        save_audio(wav, sr, "./mock_" + os.path.basename(re.sub("_i", "_o", mock_audio_fp)))
        print(type(wav), wav.shape, sr)
        #play_audio(wav.tobytes(), sr)


    if False:
        # for fname in ["pretrained-11-7-21_75k.pt", "40000.pt", "50000.pt", "60000.pt"]:
        for fname in ["pretrained-11-7-21_75k.pt"]:
            print("mock use model: %s" % fname)
            M = Text2Speech(encoder_fp="./sounda_voice_models/encoder/pretrained1.pt",
                            synth_fp="./sounda_voice_models/synth/%s" % fname,
                            vocoder_fp="./sounda_voice_models/vocoder/g_hifigan.pt")
            M.init()
            # a
            wav, sr = M.synth_file(text, mock_audio_fp)
            # print(type(wav), wav.shape, sr)
            play_audio(wav.tobytes(), sr)
            utils_audio.save_audio_buffer(wav.tobytes(), sr, "mock_audio_%s.wav" % fname)



    if False:
        M = Text2Speech(encoder_fp="./sounda_voice_models/encoder/pretrained1.pt",
                        synth_fp="./sounda_voice_models/synth/pretrained-11-7-21_75k.pt",
                        vocoder_fp="./sounda_voice_models/vocoder/g_hifigan.pt")
        M.init()
        # a
        wav, sr = M.synth_file(text, mock_audio_fp)
        print(type(wav), wav.shape, sr)
        play_audio(wav.tobytes(), sr)

        # b
        wav, sample_rate = librosa.load(mock_audio_fp)
        wav, sr = M._notwork_synth(text, wav)
        print(type(wav), wav.shape, sr)
        play_audio(wav.tobytes(), sr)

        # c
        wav, sr = synthesize(texts=text,
                             mock_audio_fp=mock_audio_fp,
                             encoder_fp="./sounda_voice_models/encoder/pretrained1.pt",
                             synth_fp="./sounda_voice_models/synth/pretrained-11-7-21_75k.pt",
                             vocoder_fp="./sounda_voice_models/vocoder/g_hifigan.pt")
        save_audio(wav, sr,  "./"+os.path.basename(re.sub("_i", "_o", mock_audio_fp)))
        print(type(wav), wav.shape, sr)
        play_audio(wav.tobytes(), sr)


