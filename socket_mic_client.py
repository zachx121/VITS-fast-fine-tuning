import time

import socketio
import pyaudio
import audioop
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import logging
from collections import deque

logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

GEN_AUDIO=False
# 创建一个Socket.IO client
sio = socketio.Client()


@sio.event(namespace='/MY_SPACE')
def get_server_info(message):
    logging.info("Server Says: '%s'\n" % message)


def play_audio(audio_buffer, sr, channels=1):
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


@sio.event(namespace='/MY_SPACE')
def audio_rsp(message):
    logging.info("[words]: %s" % message['text'])
    if GEN_AUDIO:
        sr, audio_buffer = message['sr'], message['audio_buffer']
        t = threading.Thread(target=play_audio, args=(audio_buffer, sr))
        t.start()
        t.join()


# sio.connect('http://localhost:8080/MY_SPACE')
# time.sleep(5)
# sio.emit('my_event', {'data': 'Hello, World!'}, namespace='/MY_SPACE')
# sio.emit('init', {}, namespace='/MY_SPACE')

# 以下是PyAudio参数，可能需要根据你的麦克风和系统进行修改
chunk = 1024*10  # 每次读取的音频数据的长度
sample_format = pyaudio.paInt16  # 16位深度
channels = 1  # 音频通道数
sample_rate = 44100  # 采样频率
threshold = 500  # RMS阈值，只有音量超过这个值时才会发送数据，可能需要根据实际情况调整

p = pyaudio.PyAudio()

logging.info(">>> opening audio stream...")
stream = p.open(format=sample_format,
                channels=channels,
                rate=sample_rate,
                frames_per_buffer=chunk,
                input=True)

logging.info(">>> 开始录音...")

try:
    gap_start = 0
    sending_audio = False

    while True:
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)  # 使用audioop.rms()函数计算音量
        if rms > threshold:
            sending_audio = True
            gap_start = 0
        elif sending_audio:
            if gap_start == 0:
                gap_start = time.time()
            elif time.time() - gap_start > 2:
                logging.info("Quiet for more than 2 second.")
                sending_audio = False
                gap_start = 0

        # print(rms, gap_start, sending_audio)

        if sending_audio:
            print("sending...", rms)
            # # print('发送: {!r}'.format(data))
            # audio_info = {"audio": data,
            #               "channels": channels,
            #               "sample_rate": fs,
            #               "gen_audio": GEN_AUDIO}
            # sio.emit('process', audio_info, namespace='/MY_SPACE')

except KeyboardInterrupt:
    print('停止录音')
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('断开连接')
    sio.disconnect()

