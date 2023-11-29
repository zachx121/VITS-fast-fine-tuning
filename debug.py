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
                          "ts": int(time.time()*1000)}
            sio.emit('speech2text', audio_info, namespace='/MY_SPACE')
            print("send.")
            buffer = f.readframes(bps)

def send_text2speech(host):
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
        print("messaged is: '%s'" % messaged)
        sr, audio_buffer = messaged['sr'], messaged['audio_buffer']
        sr = int(sr)
        audio_buffer = base64.b64decode(audio_buffer)
        print("receive buffer: len=%s" % len(audio_buffer))
        # 服务端目前是把32转16了（因为客户端耳机只能用16？）
        utils_audio.save_audio_buffer(audio_buffer, sr, "./vits_t2s_%s_%s.wav" % (int(time.time()), messaged["trace_id"]), dtype=np.int16)
        # t = threading.Thread(target=utils_audio.play_audio, args=(audio_buffer, sr))
        # t.start()
        # t.join()

    # host = "https://zach-0p2qy1scjuj9.serv-c1.openbayes.net"
    sio.connect(host + '/MY_SPACE')
    print("send 1st")
    sio.emit('text2speech', json.dumps({'text': "我啥都没听到", 'speaker': 'zh_m_daniel'}), namespace="/MY_SPACE")

    print("send 2nd")
    sio.emit('text2speech', json.dumps({'text': "I'm speaking english now", 'speaker': 'en_m_senapi'}), namespace="/MY_SPACE")
    all_spekers = "en_m_apple,en_m_armstrong,en_m_pengu,en_m_senapi,en_wm_Beth,en_wm_Boer,en_wm_Kathy,zh_m_AK,zh_m_daniel,zh_m_silang,zh_m_TaiWanKang,zh_wm_TaiWanYu,zh_wm_Annie"
    for i in all_spekers.split(","):
        sio.emit('text2speech', json.dumps({'text': "I'm speaking english now", 'speaker': i, 'trace_id': i}), namespace="/MY_SPACE")
    time.sleep(10)


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
            utils_audio.save_audio_buffer(audio_buffer, sr, "./vits_t2s_%s.wav" % int(time.time()), dtype=np.int16)
            # t = threading.Thread(target=utils_audio.play_audio, args=(audio_buffer, sr))
            # t.start()
            # t.join()

    # host = "http://127.0.0.1:8080"
    host = "http://region-45.autodl.pro:29369"
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

    for speaker in ["四郎配音", "daniel", "xr", "zhongli"]:
        sio.emit('text2speech',
                 json.dumps({'text': "致以诚挚的祝愿和美好的祝福", 'speaker': speaker, "language": "zh"}),
                 namespace="/MY_SPACE")
        time.sleep(1)

    time.sleep(30)
    sys.exit(0)

def send_speech2text(host):
    import os
    import wave
    import socketio
    import logging
    import librosa
    from tqdm.auto import tqdm
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

    @sio.on("speech2text_rsp", namespace="/MY_SPACE")
    def speech2text_rsp(message):
        messaged = json.loads(message)
        print("messaged is: '%s'" % messaged)

    sio.connect(host + '/MY_SPACE')

    fp = os.path.abspath("./prehot_speech2text.wav")
    # fp = os.path.abspath("/Volumes/MacData/下载HDD/zh_wm_Annie_30s.wav")
    with wave.open(fp, 'rb') as wf:
        print("声道数: %s" % wf.getnchannels())
        print("采样率: %s" % wf.getframerate()) # 一秒多少个样本（样本数*样本宽就是一秒多少字节）
        print("采样总数(样本总数、数组长度): %s" % wf.getnframes())
        print("采样宽度(样本宽度): %s" % wf.getsampwidth())  # 标准的16位PCM音频中，每个样本占用2个字节
        print("音频总时长: %s" % (wf.getnframes()/wf.getframerate()))

        standard_dtypes = {2:np.int16, 4:np.int32}
        dtype = standard_dtypes[wf.getsampwidth()]
        all_time, each_time = 0, 1.0
        while all_time < wf.getnframes()/wf.getframerate():
            chunk_size = int(wf.getframerate() * each_time)  # 采样率*采样时间拿到样本数（取整）
            buffer = wf.readframes(chunk_size)
            # 如果是双声道要处理成单声道
            if wf.getnchannels() == 2:
                audio_data = np.frombuffer(buffer, dtype=np.int16).reshape(-1, 2)
                buffer = audio_data.mean(axis=1).astype(np.int16)
            # 注意这里的逻辑：
            # 每次循环读取each_time时长的样本，对应的是chunk_size个样本
            # 每次拿到的样本字节流长度（即buffer长度）是chunk_size*sample_width
            # 所以最后一次拿到的buffer想要补零的话：
            #  - 首先计算本次实际拿到的“样本个数”是 len(buffer)//sample_width
            #  - 然后用numpy补上 “chunk_size-样本个数” 的零
            print(">>> eatch_time:%s, chunk_size: %s, buffer_len: %s, buffer2arr_shape: %s" % (each_time,chunk_size,len(buffer),np.frombuffer(buffer,dtype=dtype).shape))
            # 服务端通过添加参数也能做补零了(sample_rate,elapse,sample_width)
            # if len(buffer)//wf.getsampwidth() < chunk_size:
            #     pad_zeros = np.zeros((chunk_size-len(buffer)//wf.getsampwidth()),dtype=dtype)
            #     pad_res = np.hstack([np.frombuffer(buffer,dtype=dtype), pad_zeros])
            #     buffer = pad_res.tobytes()
            #     print("after padding a shape=%s zeros, buffer_len: %s" % (pad_zeros.shape, len(buffer)))
            # Audio(np.frombuffer(buffer, dtype=dtype), rate=wf.getframerate())
            all_time += each_time
            audio_info = {"audio": base64.b64encode(buffer).decode(),
                          "channels": wf.getnchannels(),
                          "sample_rate": wf.getframerate(),
                          "elapse": each_time,
                          "sample_width": wf.getsampwidth(),  # 注意服务端只会对位宽2的音频做补零
                          "language": "zh",
                          "ts": int(time.time()*1000)
                          }
            audio_info_json = json.dumps(audio_info)
            sio.emit('speech2text', audio_info_json, namespace='/MY_SPACE')
            time.sleep(0.5)

    time.sleep(15)
    # 断开连接
    sio.disconnect()


if __name__ == '__main__':
    print(sys.argv)
    assert len(sys.argv) >= 3
    if sys.argv[1] == "speech2text":
        send_speech2text(sys.argv[2])
    elif sys.argv[1] == "text2speech":
        send_text2speech(sys.argv[2])
    time.sleep(5)
