
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

    def init(self):
        self.model = whisper.load_model(self.model_type, download_root=self.download_root)
        return self

    def transcribe(self, audio_file):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        result = self.model.transcribe(audio_file,
                                       task="transcribe",
                                       beam_size=5,
                                       best_of=5,
                                       word_timestamps=False)
        return result['text']


print(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
M_stt = Speech2Text(model_type=MODEL_TYPE, download_root="./whisper_models").init()
time_res = timeit.timeit(lambda: M_stt.transcribe("./audio_daniel_2021-part0.wav"), number=10)
print(">>> timeit: %s" % time_res)
text = M_stt.transcribe("./audio_daniel_2021-part0.wav")
print(">>> transcribe:\n%s" % text)

# import pickle
# with open("./result_%s.pkl" % MODEL_TYPE, "wb") as f:
#     pickle.dump(result, f)

# import json
# print(json.dumps(result))



