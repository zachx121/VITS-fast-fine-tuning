import logging
import pickle
import queue
import multiprocessing as mp
import numpy as np

#mp.set_start_method("fork")
#mp.set_start_method("forkserver")
logging.basicConfig(format='[%(asctime)s-%(process)d-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
import warnings
import numba
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

import random
import whisper.tokenizer
import utils_audio
from speech2text import Speech2Text
#from text2speech import Text2Speech
# from sounda_voice.text2speech import Text2Speech
from text2speech import Text2Speech as Text2Speech_vits
from flask import Flask, render_template, request, current_app
from flask_socketio import SocketIO, emit,  join_room
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
import time
import audioop
import torch
import sys
import threading
import base64
import json
import os
sys.path.append("./")
from datetime import datetime
import wave

# [Model prepared]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(">>> [DEVICE]: %s" % DEVICE)

DEBUG = False
PORT = 6006
PROCESS_NUM = 2
TTS_MODEL_DIR = "/root/autodl-fs/vits_models/OUTPUT_MODEL_en"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(">>> [PORT]: %s" % PORT)
logging.info(">>> [TTS_MODEL_DIR]: %s" % TTS_MODEL_DIR)


NAME_SPACE = '/MY_SPACE'
SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS
AUDIO_RECORD = {}

# 子线程用vits进行声音合成（文本转语音）
def process_queue_text2speech(q_input, q_output):
    logging.info("process_queue_text2speech start.")
    logging.info(">>> Construct&Init Model (device is '%s')" % DEVICE)
    M_tts = Text2Speech_vits(model_dir=os.path.join(TTS_MODEL_DIR,"G_latest.pth"),
                             config_fp=os.path.join(TTS_MODEL_DIR,"config.json"))
    M_tts.init()
    logging.info(">>> Construct&Init Model done.")
    logging.info(">>> Speakers: %s" % ",".join(M_tts.hparams['speakers'].keys()))

    while True:
        # 从队列中获取数据
        data, t_str, sid = q_input.get()
        ts_data = int(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").timestamp())
        if data is None:
            logging.info("data is None, break now. %s" % t_str)
            break
        t_begin = time.time()
        logging.debug("  Process of sid-%s-%s start." % (sid, t_str))

        if data["speaker"] in M_tts.hparams['speakers'].keys():
            sr, wav = M_tts.tts_fn(text=data["text"],
                                   speaker=data["speaker"],
                                   language="auto",  # 用Mix的话就相当于直接读字母发音了
                                   text_cleaners=M_tts.hparams['data']['text_cleaners'],
                                   speed=1)
            # 默认得到的是np.float32，32位深的结果，这里改成16位深
            wav_scaled = np.clip(wav, -1.0, 1.0)
            wav_16bit = (wav_scaled * 32767).astype(np.int16)
            rsp = {"trace_id": data.get("trace_id",""),
                   "audio_buffer": base64.b64encode(wav_16bit.tobytes()).decode(),
                   "sr": str(sr),
                   "status": "0",
                   "msg": "success."}
            rsp = json.dumps(rsp)
            q_output.put((rsp, sid))
            _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_text2speech" % PORT)
        else:
            logging.error("not found speaker: '%s'" % data["speaker"])
            rsp = {"trace_id": data.get("trace_id",""),
                   "audio_buffer": "",
                   "sr": "",
                   "status": "1",
                   "msg": "fail. not found speaker '%s'" % data['speaker']}
            rsp = json.dumps(rsp)
            q_output.put((rsp, sid))
            _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_text2speech" % PORT)

# Flask Service init
def create_app():
    import eventlet
    eventlet.monkey_patch()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret_key'
    socketio = SocketIO()
    socketio.init_app(app, cors_allowed_origins='*', async_mode='eventlet')
    # socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')
    # socketio.init_app(app, cors_allowed_origins='*')

    # 以「127.0.0.1:8080/debug」这个页面的访问，来触发一次服务端对客户端的socket消息发送
    # curl 127.0.0.1:8080/debug
    @app.route("/debug")
    def debug_func():
        return "message send!\n"

    # 针对NAME_SPACE域里的socket连接的「connect」事件
    @socketio.on("connect", namespace=NAME_SPACE)
    def connect_msg():
        logging.info("client connected.")
        ts = int(time.time())
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        info = {
            "connect_time": ts,
            "connect_time_str": ts_str,
            "last_active_time": ts,
            "buffer": b"",
            "text": ""
        }
        # app = current_app._get_current_object()
        # thread = socketio.start_background_task(target=process_queue_text2speech, app=app)
        with LOCK:
            SID_INFO.update({request.sid: info})

    # 针对NAME_SPACE域里的socket连接的「disconnect」事件
    @socketio.on("disconnect", namespace=NAME_SPACE)
    def disconnect_msg():
        logging.info("client disconnected.")

    @socketio.on("text2speech", namespace=NAME_SPACE)
    def text2speech(data):
        # 放到子线程里做
        data = json.loads(data)
        ts = int(time.time())
        t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"{NAME_SPACE}/text2speech received an input.(%s %s)" % (ts, t_str))

        # 将数据添加到队列中
        Q_text2speech.put((data, t_str, request.sid))

    @app.route("/exec_emit_text2speech")
    def exec_emit_text2speech():
        # logging.debug("触发了向客户端发送消息（持续从队列中加载直到没有可加载的）")
        while True:
            try:
                rsp, sid = Q_text2speech_rsp.get_nowait()
                # logging.debug("    队列拿到结果 —— rsp:'%s...'" % str(rsp[:15]))
                if DEBUG:
                    with open(os.path.join(OUTPUT_DIR, "response.pkl"), "wb") as fwb:
                        pickle.dump(rsp, fwb)
                socketio.emit("text2speech_rsp", rsp, to=sid, namespace=NAME_SPACE)
            except Empty:
                # logging.debug("    队列为空，退出")
                break
        return ""

    return app, socketio


if __name__ == '__main__':
    PROCESS_NUM = int(sys.argv[1]) if len(sys.argv) >= 2 else PROCESS_NUM
    logging.info(">>> 并行进程数量: %s" % PROCESS_NUM)
    mp.set_start_method("forkserver")
    manager = mp.Manager()
    Q_text2speech = manager.Queue()
    Q_text2speech_rsp = manager.Queue()
    SID_INFO = manager.dict()
    LOCK = mp.Lock()
    _PID_NAME = manager.dict()
    _PID_NAME[os.getpid()] = "主进程"
    processes = []
    for idx in range(PROCESS_NUM):
        p2 = mp.Process(target=process_queue_text2speech,
                        args=(Q_text2speech, Q_text2speech_rsp))
        p2.start()
        processes.append(p2)
        _PID_NAME.update({p2.pid: "子进程%s" % idx})
        logging.info("    text2speech子进程启动 (%s)" % p2.pid)
        
    logging.info(">>> MultiProcess ready. all pid as follow:")
    logging.info(_PID_NAME)

    app, sio = create_app()
    logging.info("\n"*2+">>> Socket App run..."+"\n"*2)
    sio.run(app, host='0.0.0.0', port=PORT, debug=False)  # 阻塞
    # gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 server_sounda_mp:app
