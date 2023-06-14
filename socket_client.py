import sys
import socketio
import time
import logging
import pyaudio
import threading
from scipy.io.wavfile import read as wav_read

import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

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
    #sio.disconnect()


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
    sr, audio_buffer = message['sr'], message['audio_buffer']
    t = threading.Thread(target=play_audio, args=(audio_buffer, sr))
    t.start()


sio.connect('http://localhost:8080/MY_SPACE')
time.sleep(1)
# 「/MY_SPACE」命名空间中的「my_event」事件，发送消息
sio.emit('my_event', {'data': 'Hello, World!'}, namespace='/MY_SPACE')
sio.emit('init', {}, namespace='/MY_SPACE')


# 将音频流分割为三秒的片段，并暂停0.5秒
rate, data = wav_read('/Users/didi/0-Code/samples/CXM/audio_daniel_2021-part0.wav')
slice_length = 3 * rate
for i in range(0, len(data), slice_length):
    audio_part = data[i:i+slice_length]
    audio_info = {"audio": audio_part.tobytes(),
                  "channels": audio_part.shape[1],
                  "sample_rate": rate,
                  "gen_audio": True}
    sio.emit('process', audio_info, namespace='/MY_SPACE')
    logging.info("Sent 3 seconds of audio data.")
    time.sleep(0.5)


