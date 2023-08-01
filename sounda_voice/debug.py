
import os
import re
import sys
from scipy.io.wavfile import write
import numpy as np
import pyaudio
import librosa
from function.encoder import inference as encoder
from function.synthesizer.inference import Synthesizer
from function.vocoder.hifigan import inference as gan_vocoder
from pathlib import Path

def synthesize(texts,
               mock_audio_fp,
               encoder_fp="./coder/encoder/pretrained1.pt",
               synth_fp="./model/pretrained-11-7-21_75k.pt",
               vocoder_fp="./coder/vocoder/pretrained/g_hifigan.pt"
               ):
    # load function
    encoder.load_model(Path(encoder_fp))
    current_synt = Synthesizer(Path(synth_fp))
    gan_vocoder.load_model(Path(vocoder_fp))

    wav, sample_rate = librosa.load(mock_audio_fp)
    # source_spec = Synthesizer.make_spectrogram(wav)
    print("sample_rate from librosa.load is %s" % sample_rate)
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


if __name__ == '__main__':
    import pypinyin
    pypinyin.load_phrases_dict({"映射测试": [["ce4"], ["shi4"], ["ying4"], ["she4"]]}, style=pypinyin.Style.TONE3)
    text1 = "测试语音，短句测试拼音，映射测试"
    text2 = "连续语音测试特殊符号「测试」，阿拉伯数字测试1234，需要单独拼音映射吗，长句测试样例"

    text = text2
    mock_audio_fp = "/Users/didi/0-Code/Sounda/voice/input/female_p1_i.wav"
    wav, sr = synthesize(texts=text,
                         mock_audio_fp=mock_audio_fp,
                         encoder_fp="./sounda_voice_models/encoder/pretrained1.pt",
                         synth_fp="./sounda_voice_models/synth/pretrained-11-7-21_75k.pt",
                         vocoder_fp="./sounda_voice_models/vocoder/g_hifigan.pt")
    save_audio(wav, sr,  "./"+os.path.basename(re.sub("_i", "_o", mock_audio_fp)))
    play_audio(wav.tobytes(), sr)


