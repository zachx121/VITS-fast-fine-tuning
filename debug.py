import base64
import sys
import time

import numpy as np

import utils_audio
import json

# 先用english2ipa拿到国际注音的ipa，然后再把ipa注音转成romaji注音
def ipa2romaji():
    # [debug]
    import text
    import json
    from utils_audio import play_audio
    from text2speech import Text2Speech

    M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                        config_fp="./configs/finetune_speaker.json").init()
    def pronounce(symbol_inp):
        sr, audio = M_tts._symbols2audio(symbol_inp)
        play_audio(audio.tobytes(), sr)

    text.mandarin.chinese_to_romaji("这是我的炸酱面")
    text.cleaners.zh_ja_mixture_cleaners("[ZH]这是我的炸酱面[ZH]")
    text.mandarin.chinese_to_romaji("深圳")
    txt = "Hello, My name is norris. Today is a good day."
    txt_ipa = text.english.english_to_ipa(txt)
    txt_lzyipa = text.english.english_to_lazy_ipa(txt)
    txt_rmj = text.english.english_to_romaji(txt)
    with open("/Users/didi/0-Code/VITS-fast-fine-tuning/configs/finetune_speaker.json", "r") as fr:
        conf = json.load(fr)
        symbols = "".join(conf["symbols"])

    print(txt)
    print(txt_ipa)
    print(txt_lzyipa)
    print(txt_rmj)
    print("".join([i if i in conf["symbols"] else "[" + i + "]" for i in txt_rmj]))
    #sr, audio = M_tts._symbols2audio(txt_rmj)
    pronounce(txt_rmj)
    # hɛˈloʊ, maɪ neɪm ɪz ˈnɔrɪs. təˈdeɪ ɪz ə gʊd deɪ.
    pronounce("hə`loɯ→, mai→ nei→m i→z `no→ɹi→s. tə`dei→ i→z ə guɯ→d dei→.")

    # təˈdeɪ ɪz ə gʊd deɪ
    print(text.english.english_to_ipa("Today is a good day"))
    print(text.english.english_to_lazy_ipa("Today is a good day"))
    # tə↓dei iz ə gɯd dei
    print(text.english.english_to_romaji("Today is a good day"))
    # _,.!?-~…AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑

    pronounce("tə_dei→ iz ə→ gu→d dei→.")
    pronounce("tə~dei→↑ is ə~ gu→d dei→.")


def find_quiet_in_buffer():
    import audioop
    import librosa
    import utils_audio
    import pickle
    import wave
    import numpy as np
    import matplotlib.pyplot as plt
    SAMPLE_RATE = 16000  # 采样频率
    SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
    CHANNELS = 1  # 音频通道数
    CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
    BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS

    def wave_way():
        wav_fp = "/Users/didi/0-Code/VITS-fast-fine-tuning/output_sid_JFlDIMDmgflvOXfBAAAB.wav"
        wav_fp = "/Users/didi/0-Code/VITS-fast-fine-tuning/output_mic_v2.wav"
        # wav_fp = "/Users/didi/0-Code/VITS-fast-fine-tuning/prehot_speech2text.wav"
        # wav_fp = "/Users/didi/0-Code/#samples/CXM/audio_daniel_2021-part0.wav"
        with wave.open(wav_fp) as f:
            audio_buffer = f.readframes(f.getnframes())
            sr = f.getframerate()
            sw = f.getsampwidth()
            c = f.getnchannels()
        print("音频总时长: %s" % (len(audio_buffer)/(sr*sw*c)))

        print("声道数: %s" % wave.open(wav_fp).getnchannels())
        print("采样率: %s" % wave.open(wav_fp).getframerate())
        print("采样总数(样本总数、数组长度): %s" % wave.open(wav_fp).getnframes())
        print("采样宽度(样本宽度): %s" % wave.open(wav_fp).getsampwidth())  # 标准的16位PCM音频中，每个样本占用2个字节
        print("音频总时长: %s" % (wave.open(wav_fp).getnframes() / wave.open(wav_fp).getframerate()))

        _audio, _sr = librosa.load(wav_fp, sr=sr)
        utils_audio.play_audio_buffer_with_volume(audio_buffer, sr, c)

    with open("/Users/didi/0-Code/VITS-fast-fine-tuning/debug.pkl", "rb") as frb:
        info = pickle.load(frb)
    info = info["TfqWukDnY0Oj24fEAAAH"]
    audio_buffer = info["buffer"]
    total_time = len(audio_buffer)/BYTES_PER_SEC

    # 结束前小于阈值的声音持续了多久
    holder = 500
    last = 0
    audio_buffer_reverse = audio_buffer[::-1]
    for i in range(0, int(total_time/0.1)):
        s, e = i*0.1*BYTES_PER_SEC, (i+1)*0.1*BYTES_PER_SEC
        s, e = int(s), int(e)
        if audioop.rms(audio_buffer_reverse[s:e], SAMPLE_WIDTH) <= holder:
            last += 0.1*1
        else:
            break

    # 折线图绘制一下audio_buffer整个时间轴上的音量
    Y,X = utils_audio.cal_rms(audio_buffer, delta=0.1)
    plt.plot(X, Y)
    plt.show()


def get_file_as_buffer_stream():
    SAMPLE_RATE = 16000  # 采样频率
    SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
    CHANNELS = 1  # 音频通道数
    CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
    BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS

    import time
    import wave
    import socketio
    sio = socketio.Client()
    host = "http://127.0.0.1:8080"
    # host = "https://zach-0p2qy1scjuj9.serv-c1.openbayes.net"
    sio.connect(host + '/MY_SPACE')

    with wave.open("/Users/didi/0-Code/#samples/CXM/audio_daniel_2021-part0.wav") as f:
        sr = f.getframerate()
        sw = f.getsampwidth()
        c = f.getnchannels()
        bps = sr*sw*c

        buffer = f.readframes(bps)
        while len(buffer) > 0:
            audio_info = {"audio": buffer,
                          "channels": CHANNELS,
                          "sample_rate": SAMPLE_RATE,
                          "ts": int(time.time())}
            sio.emit('speech2text', audio_info, namespace='/MY_SPACE')
            print("send.")
            buffer = f.readframes(bps)

def send_text2speech():
    import socketio
    import logging
    import threading
    import utils_audio
    sio = socketio.Client()

    @sio.event(namespace='/MY_SPACE')
    def connect():
        logging.info('connection established')

    @sio.event(namespace='/MY_SPACE')
    def disconnect():
        logging.info('disconnected from server')

    @sio.event(namespace='/MY_SPACE')
    def get_server_info(message):
        logging.info("Server Says: '%s'\n" % message)
        if message == "All init done.":
            logging.info(">>> 服务端已就绪，可以开始说话...")

    @sio.on("text2speech_rsp", namespace="/MY_SPACE")
    def text2speech_rsp(message):
        messaged = json.loads(message)
        sr, audio_buffer = messaged['sr'], messaged['audio_buffer']
        sr = int(sr)
        audio_buffer = base64.b64decode(audio_buffer)
        print("receive buffer: len=%s" % len(audio_buffer))
        utils_audio.save_audio_buffer(audio_buffer, sr, "./vits_t2s_%s.wav" % int(time.time()))
        t = threading.Thread(target=utils_audio.play_audio, args=(audio_buffer, sr))
        t.start()
        t.join()

    host = "http://127.0.0.1:8080"
    # host = "https://zach-0p2qy1scjuj9.serv-c1.openbayes.net"
    sio.connect(host + '/MY_SPACE')
    sio.emit('my_event', {'data': 'Hello, World!'}, namespace='/MY_SPACE')
    print("send..")
    sio.emit('text2speech', json.dumps({'text': "我啥都没听到"}), namespace="/MY_SPACE")
    time.sleep(1)
    print("send..")
    time.sleep(1)
    sio.emit('text2speech', json.dumps({'text': "我是谁"}), namespace="/MY_SPACE")


def send_text2speech_sounda():
    import librosa
    import socketio
    import logging
    import threading
    import utils_audio
    sio = socketio.Client()

    @sio.event(namespace='/MY_SPACE')
    def connect():
        logging.info('connection established')

    @sio.event(namespace='/MY_SPACE')
    def disconnect():
        logging.info('disconnected from server')

    @sio.on("upload_speaker_rsp", namespace="/MY_SPACE")
    def upload_speaker_rsp(message):
        print(upload_speaker_rsp)

    @sio.on("text2speech_rsp", namespace="/MY_SPACE")
    def text2speech_rsp(message):
        messaged = json.loads(message)
        if messaged["status"] != "0":
            logging.error("Some Err. message is '%s'" % messaged["msg"])
        else:
            sr, audio_buffer = messaged['sr'], messaged['audio_buffer']
            sr = int(sr)
            audio_buffer = base64.b64decode(audio_buffer)
            print("receive buffer: len=%s" % len(audio_buffer))
            utils_audio.save_audio_buffer(audio_buffer, sr, "./vits_t2s_%s.wav" % int(time.time()))
            t = threading.Thread(target=utils_audio.play_audio, args=(audio_buffer, sr))
            t.start()
            t.join()

    host = "http://127.0.0.1:8080"
    # host = "https://zach-0p2qy1scjuj9.serv-c1.openbayes.net"
    sio.connect(host + '/MY_SPACE')

    # 上传需要克隆的声音素材
    def upload():
        print("uploading...")
        wav, source_sr = librosa.load("/Users/didi/0-Code/#samples/CXM/audio_daniel_2021-part0.wav", sr=None)
        upd_speaker = {"audio": base64.b64encode(wav.tobytes()).decode(),
                       "speaker": "cxm",
                       "language": "zh"}
        sio.emit('upload_speaker', json.dumps(upd_speaker), namespace='/MY_SPACE')
        time.sleep(10)

        print("uploading...")
        wav, source_sr = librosa.load("/Users/didi/0-Code/#samples/female_p1_i.wav", sr=None)
        upd_speaker = {"audio": base64.b64encode(wav.tobytes()).decode(),
                       "speaker": "female_p1",
                       "language": "zh"}
        sio.emit('upload_speaker', json.dumps(upd_speaker), namespace='/MY_SPACE')
        time.sleep(10)

        print("uploading...")
        wav, source_sr = librosa.load("/Users/didi/0-Code/#samples/male_p1_i.wav", sr=None)
        upd_speaker = {"audio": base64.b64encode(wav.tobytes()).decode(),
                       "speaker": "male_p1",
                       "language": "zh"}
        sio.emit('upload_speaker', json.dumps(upd_speaker), namespace='/MY_SPACE')
        time.sleep(10)

    #upload()

    print("send..")
    sio.emit('text2speech', json.dumps({'text': "我啥都没听到", 'speaker':"cxm", "language":"zh"}), namespace="/MY_SPACE")
    time.sleep(5)
    print("send..")
    sio.emit('text2speech', json.dumps({'text': "我啥都没听到", 'speaker':"female_p1", "language":"zh"}), namespace="/MY_SPACE")
    time.sleep(5)
    print("send..")
    sio.emit('text2speech', json.dumps({'text': "我啥都没听到", 'speaker':"male_p1", "language":"zh"}), namespace="/MY_SPACE")
    time.sleep(5)
    sys.exit(0)


if __name__ == '__main__':
    send_text2speech_sounda()
