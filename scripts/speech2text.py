
import whisper

# tiny,base,small,medium,large-v1,large-v2
# VITS用的是medium
import whisper
MODEL_TYPE = "tiny"
model = whisper.load_model(MODEL_TYPE, download_root="./whisper_models")
options = dict(beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)
result = model.transcribe("./audio_daniel_2021-part0.wav",
                          word_timestamps=True, **transcribe_options)
print(result["text"])

import pickle
with open("./result_%s.pkl" % MODEL_TYPE, "wb") as f:
    pickle.dump(result, f)

# import json
# print(json.dumps(result))



