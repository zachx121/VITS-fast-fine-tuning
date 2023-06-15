import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s-SERVER]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from speech2text import Speech2Text
from text2speech import Text2Speech
from flask import Flask, render_template
from flask_socketio import SocketIO, emit,  join_room
import numpy as np
from collections import deque
import time
import audioop
import torch
from scipy.io.wavfile import write as write_wav
# [Flask Service init]
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*')
NAME_SPACE = '/MY_SPACE'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(">>> Construct Model (device is '%s')" % DEVICE)
M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                    config_fp="./configs/finetune_speaker.json",
                    device=DEVICE)
M_stt = Speech2Text(model_type="tiny", download_root="./whisper_models")

logging.info(">>> Construct Model done.")

# 用网页的js作为socket的客户端
@app.route("/")
def index():
    return render_template("index.html")


# 以「127.0.0.1:8080/push」这个页面的访问，来触发一次服务端对客户端的socket消息发送
@app.route("/push")
def push_once():
    logging.info("will send to clinet.")
    socketio.emit("dcenter", {"data": "this is a test message"}, namespace=NAME_SPACE)
    return "message send!"


# 针对NAME_SPACE域里的socket连接的「connect」事件
@socketio.on("connect", namespace=NAME_SPACE)
def connect_msg():
    logging.info("client connected.")


# 针对NAME_SPACE域里的socket连接的「disconnect」事件
@socketio.on("disconnect", namespace=NAME_SPACE)
def disconnect_msg():
    logging.info("client disconnected.")


@socketio.on("my_event", namespace=NAME_SPACE)
def mtest_message(message):
    logging.info("client's message as follow: %s" % message)
    emit("my_response", {"data": "the server-side has received your message as follow:\n '%s'" % message})
    emit("get_server_info", {"data": "server send u message in event:get_server_info"})
    logging.info("send message to client.")


def _process_audio(data):
    text = M_stt.transcribe_buffer(data['audio'],
                                   sr_inp=data['sample_rate'],
                                   channels_inp=data['channels'],
                                   fp16=False)
    if data.get("gen_audio", False):
        sr, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
    else:
        sr = 0.0
        audio = np.array(0)
    rsp = {"text": text, "audio_buffer": audio.tobytes(), "sr": sr}
    return rsp

buffer_cache = deque()
buffer_duration = 0.0
last_volume_check = 0
low_volume_duration = 0.0
VOLUME_THRESHOLD = 500  # 你需要设定一个阈值

@socketio.on("process_v2", namespace=NAME_SPACE)
def process_audio_v2(data):
    global buffer_duration, last_volume_check, low_volume_duration
    logging.info(f"{NAME_SPACE}_audio received an input.")
    # 计算此次音频buffer的时长，这需要你知道音频的采样率和channels
    # 这里x2是因为1.data['audio']是一个字节流，2.标准的16位PCM音频中，每个样本占用2个字节
    buffer_duration += len(data['audio']) / (2 * data['sample_rate'] * data['channels'])

    # 检查音量是否低于阈值，这需要一个函数来计算音量
    if audioop.rms(data['audio'], 2) < VOLUME_THRESHOLD:
        if last_volume_check == 0:
            last_volume_check = time.time()
        low_volume_duration += time.time() - last_volume_check
    else:
        # 出现大于阈值的音量，就重置low_volume_duration
        low_volume_duration = 0.0
        last_volume_check = 0
    buffer_cache.append(data['audio'])
    print(buffer_duration, last_volume_check, low_volume_duration)
    if buffer_duration >= 5.0 or low_volume_duration >= 2.0:
        # 如果buffer时长超过5秒，或者音量持续2秒低于阈值，则转录音频
        audio_buffer = b''.join(buffer_cache)
        # rsp = _process_audio({**data, **{"audio": audio_buffer}})
        # emit("audio_rsp", rsp)
        print("buffer时长超过5秒，或者音量持续2秒低于阈值，则转录音频")
        # 清空缓存和计时器
        buffer_cache.clear()
        buffer_duration = 0.0
        low_volume_duration = 0.0
        last_volume_check = 0


import queue
import threading
# 创建一个线程安全的队列
data_queue = queue.Queue()

def process_queue():
    while True:
        # 从队列中获取数据
        data = data_queue.get()
        if data is None:
            break

        # 处理数据
        text = M_stt.transcribe_buffer(data['audio'],
                                       sr_inp=data['sample_rate'],
                                       channels_inp=data['channels'],
                                       fp16=False)
        if data.get("gen_audio", False):
            sr, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
        else:
            sr = 0.0
            audio = np.array(0)
        rsp = {"text": text, "audio_buffer": audio.tobytes(), "sr": sr}

        # 发送回应
        emit("audio_rsp", rsp)

        # 标记这个任务为完成
        data_queue.task_done()

# 创建并启动一个新线程来处理队列中的数据
threading.Thread(target=process_queue, daemon=True).start()


@socketio.on("process", namespace=NAME_SPACE)
def process_audio(data):
    logging.info(f"{NAME_SPACE}_audio received an input.")

    # 将数据添加到队列中
    data_queue.put(data)


@socketio.on("init", namespace=NAME_SPACE)
def init_model(data):
    M_tts.init()
    logging.info(">>> M_tts init done.")
    M_stt.init()
    logging.info(">>> M_stt init done.")
    logging.info(">>> All init done.")
    emit("get_server_info", "All init done.")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)



# [VITS Stuff]
# from speech2text import Speech2Text
# from text2speech import Text2Speech
# M_stt = Speech2Text(model_type="medium", download_root="./whisper_models").init()
# M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
#                     config_fp="./configs/finetune_speaker.json").init()

# def socket_accept_audio(data):
#     #todo. 接受客户端的音频数据buffer
#     audio_buffer = data['audio']
#     audio_sr = data['sample_rate']
#     ###
#     text = M_stt.transcribe_data(audio_buffer, audio_sr)
#     sample_rate, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
#     ###
#     #todo. 把采样率和audio返回给客户端，客户端应该也是需要采样率sample_rate和声音数组audio才能播放的
#     socketio.emit("event", audio, sample_rate)
