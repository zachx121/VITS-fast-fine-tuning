import logging
import random

logging.basicConfig(format='[%(asctime)s-%(levelname)s-SERVER]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
from speech2text import Speech2Text
from text2speech import Text2Speech
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit,  join_room
from flask import copy_current_request_context
import numpy as np
from collections import deque
import time
import audioop
import torch
import sys
from scipy.io.wavfile import write as write_wav
from concurrent.futures import ThreadPoolExecutor

# [Params]
PORT = int(sys.argv[1]) if len(sys.argv) >= 2 else 8080
TTS_MODEL = sys.argv[2] if len(sys.argv) >= 3 else "./vits_models/G_latest_cxm_1st.pth"
SST_MODEL_DIR = sys.argv[3] if len(sys.argv) >= 4 else "./whisper_models"
logging.info(">>> [PORT]: %s" % PORT)
logging.info(">>> [TTS_MODEL]: %s" % TTS_MODEL)
logging.info(">>> [SST_MODEL_DIR]: %s" % SST_MODEL_DIR)

# [Model prepared]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(">>> Construct Model (device is '%s')" % DEVICE)
M_tts = Text2Speech(model_dir=TTS_MODEL,
                    config_fp="./configs/finetune_speaker.json",
                    device=DEVICE)
M_stt = Speech2Text(model_type="tiny", download_root=SST_MODEL_DIR, device=DEVICE)

logging.info(">>> Construct Model done.")


def init_model(*args, **kwargs):
    if not M_tts.is_init:
        M_tts.init()
        logging.info(">>> M_tts init done.")
    if not M_stt.is_init:
        M_stt.init()
        logging.info(">>> M_stt init done.")
    logging.info(">>> All init done.")


init_model()

# [Flask Service init]
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*', async_mode='eventlet')
import eventlet
eventlet.monkey_patch()
# socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')
NAME_SPACE = '/MY_SPACE'



# 用网页的js作为socket的客户端
@app.route("/")
def index():
    return render_template("index.html")


# 以「127.0.0.1:8080/push」这个页面的访问，来触发一次服务端对客户端的socket消息发送
@app.route("/push")
def push_once():
    logging.info("will send to clinet.")
    socketio.emit("get_server_info", {"data": "this is a test message"}, namespace=NAME_SPACE)
    socketio.emit("audio_rsp", {"text": "manually debug send."}, namespace=NAME_SPACE)

    return "message send!"


# 针对NAME_SPACE域里的socket连接的「connect」事件
@socketio.on("connect", namespace=NAME_SPACE)
def connect_msg():
    logging.info("client connected.")


# 针对NAME_SPACE域里的socket连接的「disconnect」事件
@socketio.on("disconnect", namespace=NAME_SPACE)
def disconnect_msg():
    logging.info("client disconnected.")

from datetime import datetime
import queue
import threading
# 创建一个线程安全的队列
data_queue = queue.Queue()
# 创建一个线程池
# executor = ThreadPoolExecutor(max_workers=5)
def process_queue():
    logging.info("process_queue start.")
    while True:
        # 从队列中获取数据
        #logging.info("process_queue start | while True start.")
        data, t_str, sid = data_queue.get()
        #logging.info("data is %s" % data)
        if data is None:
            logging.info("data is None, break now. %s" % t_str)
            break

        t_begin = time.time()
        logging.debug("  Process of sid-%s-%s start." % (sid, t_str))
        # 处理数据
        text = M_stt.transcribe_buffer(data['audio'],
                                       sr_inp=data['sample_rate'],
                                       channels_inp=data['channels'],
                                       fp16=False)
        logging.debug("  transcribed: '%s'" % text)
        if data.get("gen_audio", False):
            logging.debug("  executing tts...")
            sr, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
        else:
            sr = 0.0
            audio = np.array(0)
        rsp = {"text": text, "audio_buffer": audio.tobytes(), "sr": sr}

        # # 模拟耗时和处理结果
        # time.sleep(random.randint(1, 10))
        # rsp = {"text": "pretend processed."}
        # 标记这个任务为完成
        data_queue.task_done()
        logging.debug("  Process of sid-%s-%s finished.(elapsed %s)" % (sid, t_str, time.time()-t_begin))
        # # 用 copy_current_request_context 也不行，因为process事件在入队后就结束了，会报错如下：
        # # - RuntimeError: 'copy_current_request_context' can only be used when a request context is active,
        # # - such as in a view function.
        # 直接把发信息来的那个客户端记下来，然后单独用socketio.emit发
        socketio.emit("audio_rsp", rsp, to=sid, namespace=NAME_SPACE)
        logging.debug("size of data_queue: %s" % data_queue.qsize())


# 创建并启动一个新线程来处理队列中的数据
#threading.Thread(target=process_queue, daemon=True).start()
socketio.start_background_task(target=process_queue)

@socketio.on("process", namespace=NAME_SPACE)
def process_audio(data):
    ts = int(time.time())
    t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{NAME_SPACE}_audio received an input.(%s)" % ts)

    # text = M_stt.transcribe_buffer(data['audio'],
    #                                sr_inp=data['sample_rate'],
    #                                channels_inp=data['channels'],
    #                                fp16=False)
    # logging.debug("  transcribed: '%s'" % text)

    # 将数据添加到队列中
    data_queue.put((data, t_str, request.sid))
    logging.debug("size of data_queue: %s" % data_queue.qsize())
    #
    # debug
    # socketio.emit好像总是会丢失？客户端除了第一条好像都没收到
    # 经检测这个是正常的
    #socketio.emit("audio_rsp", {"text": "server-side copy. send a signal back (%s)" % t_str}, to=request.sid, namespace=NAME_SPACE)
    #logging.debug("send a manually response to client.")


@socketio.on("init", namespace=NAME_SPACE)
def init(*args, **kwargs):
    init_model(*args, **kwargs)
    emit("get_server_info", "All init done.")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)



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
