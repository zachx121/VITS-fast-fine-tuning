import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
import whisper

# tiny,base,small,medium,large-v1,large-v2
# VITS用的是medium
import whisper
import sys
import timeit

MODEL_TYPE = sys.argv[1] if len(sys.argv) >= 2 else "tiny"


class Speech2Text:
    def __init__(self, model_type, download_root="./whisper_models"):
        self.model_type = model_type
        self.download_root = download_root
        self.model = None

    def init(self, prehot_audio="./prehot_speech2text.wav"):
        self.model = whisper.load_model(self.model_type, download_root=self.download_root)
        try:
            self.transcribe(prehot_audio)
        except:
            logging.warning("pre_hot fail.")
        return self

    def transcribe(self, audio_file):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        result = self.model.transcribe(audio_file,
                                       task="transcribe",
                                       beam_size=5,
                                       best_of=5,
                                       word_timestamps=False)
        return result['text']


logging.info(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
M_stt = Speech2Text(model_type=MODEL_TYPE, download_root="./whisper_models").init()
text = M_stt.transcribe("./audio_daniel_2021-part0.wav")
logging.info(">>> transcribe:\n%s" % text)

# timeit
time_res = timeit.timeit(lambda: M_stt.transcribe("./audio_daniel_2021-part0.wav"), number=3)
logging.info(">>> timeit: %.4f" % (time_res/3.0))

# import pickle
# with open("./result_%s.pkl" % MODEL_TYPE, "wb") as f:
#     pickle.dump(result, f)

# import json
# print(json.dumps(result))



