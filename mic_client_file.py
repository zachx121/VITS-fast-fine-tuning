import time

import socketio
import wave
import numpy as np
import base64
import json
import librosa
import os
import logging

NAME_SPACE = '/MY_SPACE'
SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS

# 初始化 socket.io 客户端
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
    logging.info("       [tmp-words]: %s (mid:%s)" % (messaged['text'], messaged['mid']))
    if messaged["mid"] == "0":
        logging.info("[words]: %s" % messaged['text'])
    # if GEN_AUDIO:
    #     sr, audio_buffer = message['sr'], message['audio_buffer']
    #     t = threading.Thread(target=play_audio, args=(audio_buffer, sr))
    #     t.start()
    #     t.join()


# host = "http://127.0.0.1:8080"
# host = "https://zach-0p2qy1scjuj9.serv-c1.openbayes.net"
host = "https://u212392-8d2a-c21cde27.beijinga.seetacloud.com/"
sio.connect(host + NAME_SPACE)
time.sleep(5)

# 音频文件路径
audio_file = os.path.abspath("./prehot_speech2text_2.wav")
print(">>> USE FILE: %s" % audio_file)

# 打开音频文件
with wave.open(audio_file, 'rb') as wf:
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    print("framerate: %s n_frames:%s" % (framerate, n_frames))
    # 计算500毫秒内的帧数
    chunk_size = int(framerate * 1)  # 500 ms

    audio2write = b""
    for _ in tqdm(range(0, n_frames, chunk_size)):
        #audio_data = wf.readframes(chunk_size)
        audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_bytes = audio_array.tobytes()

        audio_info = {"audio": base64.b64encode(audio_bytes).decode(),
                      "channels": CHANNELS,
                      "sample_rate": SAMPLE_RATE,
                      "language": "zh",
                      "ts": int(time.time())
                      }
        audio_info_json = json.dumps(audio_info)
        sio.emit('speech2text', audio_info_json, namespace=NAME_SPACE)
        # print(audio_info["audio"])
        audio2write += b""+base64.b64decode(audio_info["audio"])

    # 发送的所有音频片段存起来
    wf_sid = wave.open("./tmp_send_audio.wav", "wb")
    wf_sid.setnchannels(CHANNELS)
    wf_sid.setsampwidth(SAMPLE_WIDTH)
    wf_sid.setframerate(SAMPLE_RATE)
    wf_sid.writeframes(audio2write)

    time.sleep(60*3)
    # 断开连接
    sio.disconnect()

