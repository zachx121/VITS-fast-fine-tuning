import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
#import gradio as gr
import librosa
import webbrowser

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn

class args:
    model_dir = "./G_latest_xr_2nd.pth"
    config_dir = "./configs/finetune_speaker.json"
    share = False

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    # parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    # parser.add_argument("--share", default=False, help="make link public (used in colab)")
    # args = parser.parse_args()
    text2speak = "叫啥，北京炸酱面？你住的实在鸟巢北苑那边吧？我到酒店休息下，卧铺没睡好，12点起来吃饭。我住在秋果酒店。我昨天在家休息了一天，下午开始搞。先把需求整理出来，还有公司注册啥的，让其他人先动起来，然后找那个人要声音demo。gpt那个测试可能得明天开始搞"
    text2speak_list = [
        "谁看这个热线的反映问题",
        "沧州一城枫景社区"
                       ]
    hps = utils.get_hparams_from_file(args.config_dir)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model_dir, net_g, None)
    logging.info(">>> hps.speakers: %s" % hps.speakers)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    for idx, text2speak in enumerate(text2speak_list):
        speaker_i = 'audio'
        logging.info(">>> processing speaker: %s" % speaker_i)
        _, audio_output = tts_fn(text=text2speak, speaker=speaker_i, language="简体中文", speed=1.0)
        sample_r, audio = audio_output
        from scipy.io.wavfile import write as write_wav
        # save audio
        filepath = "output_audio_local_%s.wav" % idx
        write_wav(filepath, sample_r, audio)

    # for i in speakers:
    #     logging.info(">>> processing speaker: %s" % i)
    #     _, audio_output = tts_fn(text=text2speak, speaker=i, language="简体中文", speed=1.0)
    #     sample_r, audio = audio_output
    #     from scipy.io.wavfile import write as write_wav
    #     # save audio
    #     filepath = "output_audio_local_%s.wav" % i
    #     write_wav(filepath, sample_r, audio)
