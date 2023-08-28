import sys
import re
from utils_audio import play_audio
from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
import utils
import commons
from text import text_to_sequence
from scipy.io.wavfile import write as write_wav
import pyaudio
import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Text2Speech:
    language_marks = {
        "Japanese": "",
        "日本語": "[JA]",
        "简体中文": "[ZH]",
        "English": "[EN]",
        "Mix": "",
    }
    lang = ['日本語', '简体中文', 'English', 'Mix']

    def __init__(self, model_dir, config_fp, device=DEVICE):
        self.model_dir = model_dir
        self.hparams = utils.get_hparams_from_file(config_fp)
        self.speaker2id = self.hparams.speakers
        self.model = None
        self.device = device
        self.is_init = False

    def init(self):
        net = SynthesizerTrn(
            len(self.hparams.symbols),
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            n_speakers=self.hparams.data.n_speakers,
            **self.hparams.model)
        _ = net.eval()
        _ = utils.load_checkpoint(self.model_dir, net, None)
        self.model = net.to(self.device)
        self.is_init = True
        return self

    # is_symbol 表示输入的text是否直接就是注音符号
    def get_text(self, text, is_symbol, text_cleaners=None):
        # cjke_cleaners
        use_cleaners = text_cleaners if text_cleaners is not None else self.hparams.data.text_cleaners
        text_norm = text_to_sequence(text,
                                     self.hparams.symbols,
                                     [] if is_symbol else use_cleaners)
        if self.hparams.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    @staticmethod
    def __mark(text):
        # 将连续的英文字符用[EN]包围 | 必须先执行英文标记
        en_pattern = re.compile(r'[a-zA-Z\s]+')
        text = re.sub(en_pattern, r'[EN]\g<0>[EN]', text)

        # 将连续的中文字符和阿拉伯数字用[ZH]包围
        zh_num_pattern = re.compile(r'[\u4e00-\u9fa5\d]+')
        text = re.sub(zh_num_pattern, r'[ZH]\g<0>[ZH]', text)

        return text

    # noise_scale 感情变化程度
    # noise_scale_w 音素发音长度
    def tts_fn(self, text, speaker, language,
               speed=1.0,
               noise_scale=0.667,
               noise_scale_w=0.8,
               text_cleaners=None):
        text_cleaners = ["zh_ja_en_mixture_cleaners"] if text_cleaners is None else text_cleaners
        if language is not None:
            if language == "auto":
                text = self.__mark(text)
            else:
                text = self.language_marks[language] + text + self.language_marks[language]
        logging.debug(f"marked text is '{text}'")
        speaker_id = self.speaker2id[speaker]
        logging.debug(f" use speaker:{speaker} speaker_id:{speaker_id}")
        stn_tst = self.get_text(text, False, text_cleaners)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            sid = LongTensor([speaker_id]).to(self.device)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid,
                                     noise_scale=noise_scale, 
                                     noise_scale_w=noise_scale_w,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return self.hparams.data.sampling_rate, audio

    def _symbols2audio(self, symbol_txt, speaker_id=0, speed=1.0):
        print("直接读注音: '%s'" % symbol_txt)
        stn_tst = self.get_text(symbol_txt, True)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            sid = LongTensor([speaker_id]).to(self.device)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid,
                                     noise_scale=.667, noise_scale_w=0.8,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        return self.hparams.data.sampling_rate, audio

    def gen_wav(self, text, output_fp, speaker="audio", language="简体中文", speed=1.0):
        sample_rate, audio = self.tts_fn(text, speaker, language, speed)
        write_wav(output_fp, sample_rate, audio)


if __name__ == '__main__':
    M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                        config_fp="./configs/finetune_speaker.json").init()

    # M_tts.gen_wav(text="叫啥，北京炸酱面？你住的是在鸟巢北苑那边吧？",
    #               speaker="audio",
    #               language="简体中文",
    #               speed=1.0,
    #               output_fp="output_audio_local.wav")

    # sr, audio = M_tts.tts_fn(text="Hello, My name is norris, I'm from Shenzhen China.",
    #                          speaker="audio",
    #                          language="English",
    #                          speed=0.7)
    # play_audio(audio.tobytes(), sr)

    sr, audio = M_tts.tts_fn(text="我是程晓敏1234。Hello, My name is norris. Today is a good day. I like China.",
                             speaker="audio",
                             language="auto",  # 用Mix的话就相当于直接读字母发音了
                             text_cleaners=["zh_ja_en_mixture_cleaners"],
                             speed=1.0)
    play_audio(audio.tobytes(), sr)

    sr, audio = M_tts.tts_fn(text="这是我的炸酱面abcd",
                             speaker="audio",
                             language="简体中文",
                             speed=0.7)
    play_audio(audio.tobytes(), sr)

    sr, audio = M_tts.tts_fn(text="这是我的炸酱面",
                             speaker="audio",
                             language="简体中文",
                             speed=0.7)
    play_audio(audio.tobytes(), sr)
    sys.exit(0)

