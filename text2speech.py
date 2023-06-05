from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
import utils
import commons
from text import text_to_sequence
from scipy.io.wavfile import write as write_wav
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

    def __init__(self, model_dir, config_fp):
        self.model_dir = model_dir
        self.hparams = utils.get_hparams_from_file(config_fp)
        self.speaker2id = self.hparams.speakers
        self.model = None

    def init(self):
        net = SynthesizerTrn(
            len(self.hparams.symbols),
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            n_speakers=self.hparams.data.n_speakers,
            **self.hparams.model)
        _ = net.eval()
        _ = utils.load_checkpoint(self.model_dir, net, None)
        self.model = net
        return self

    def get_text(self, text, is_symbol):
        text_norm = text_to_sequence(text,
                                     self.hparams.symbols,
                                     [] if is_symbol else self.hparams.data.text_cleaners)
        if self.hparams.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def tts_fn(self, text, speaker, language, speed):
        if language is not None:
            text = self.language_marks[language] + text + self.language_marks[language]
        speaker_id = self.speaker2id[speaker]
        stn_tst = self.get_text(text, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(DEVICE)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(DEVICE)
            sid = LongTensor([speaker_id]).to(DEVICE)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid,
                                     noise_scale=.667, noise_scale_w=0.8,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return self.hparams.data.sampling_rate, audio

    def gen_wav(self, text, speaker, language, speed, output_fp):
        sample_rate, audio = self.tts_fn(text, speaker, language, speed)
        write_wav(output_fp, sample_rate, audio)


M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                    config_fp="./configs/finetune_speaker.json").init()
M_tts.gen_wav(text="叫啥，北京炸酱面？你住的是在鸟巢北苑那边吧？",
              speaker="audio",
              language="简体中文",
              speed=1.0,
              output_fp="output_audio_local.wav")
