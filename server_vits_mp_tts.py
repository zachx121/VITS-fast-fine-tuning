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


# [Params]
# PORT = int(sys.argv[1]) if len(sys.argv) >= 2 else 8080
# TTS_MODEL = sys.argv[2] if len(sys.argv) >= 3 else "./vits_models/G_latest_cxm_1st.pth"
# TTS_MODEL_DIR = sys.argv[2] if len(sys.argv) >= 3 else "./sounda_voice_model_v1"
# SST_MODEL_DIR = sys.argv[3] if len(sys.argv) >= 4 else "./whisper_models"
PORT = 6006
PROCESS_NUM = 2
TTS_MODEL_DIR = "/root/autodl-fs/VITS_DATA/OUTPUT_MODEL_四郎_daniel_bruce"
SST_MODEL_DIR = "/root/autodl-fs/whisper"
WHISPER_MODEL = "large-v2"  # large-v2, tiny, medium
OUTPUT_DIR = "./output"
VOICE_SAMPLE_DIR = "./voice_sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
logging.info(">>> [PORT]: %s" % PORT)
logging.info(">>> [TTS_MODEL_DIR]: %s" % TTS_MODEL_DIR)
logging.info(">>> [SST_MODEL_DIR]: %s" % SST_MODEL_DIR)


# [Model prepared]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NAME_SPACE = '/MY_SPACE'
SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS
RMS_LAST_TIME = 0.5  # 取音频的最后多久计算RMS
RMS_HOLDER = 777  # 最后一段音频小于多少音量时，视为结束，清空buffer
DEBUG = True
AUDIO_RECORD = {}

# 子进程用whisper处理语音转文本，并将结果写入rsp队列
def process_queue_speech2text(q_input, q_output, sid_info, lock, _pid_name):
    # logging.info(_PID_NAME[os.getpid()]+"process_queue_speech2text start.")
    logging.info("process_queue_speech2text start.")  # 这个时候还没while阻塞，主进程还没执行到更新_PID_NAME字典拿不到NAME
    logging.debug(">>> Construct&Init Model (device is '%s')" % DEVICE)
    model = Speech2Text(model_type=WHISPER_MODEL, download_root=SST_MODEL_DIR, device=DEVICE)
    model.init()
    logging.debug(">>> Construct&Init Model done.")
    logging.debug("process_queue_speech2text ready.")

    while True:
        # 从队列中获取数据
        data, t_str, sid = q_input.get()
        ts_data = data['ts']
        lang = data.get("language", None)
        lang = lang if lang in whisper.tokenizer.LANGUAGES else None

        if data is None:
            logging.info(_pid_name[os.getpid()] + "data is None, break now. %s" % t_str)
            break

        t_begin = time.time()
        logging.debug(_pid_name[os.getpid()] + "Process of sid-%s-%s start." % (sid, t_str))

        eos_tag = False
        # 处理数据，这里每个sid只能交由同一个子进程处理，避免如下情况
        # 00:01 进程A在锁内追加buffer，然后即将要在锁外执行transcribe_buffer
        # 00:02 进程B在锁内判定音频结束然后清空了buffer，
        with lock:
            if sid in sid_info:
                info = sid_info[sid]
                info["buffer"] += data['audio']
                info["last_active_time"] = ts_data
                # 根据最后一秒的音量判断当前是不是eos
                lb = len(info["buffer"])
                volume = audioop.rms(info["buffer"][lb - int(RMS_LAST_TIME * BYTES_PER_SEC):lb], SAMPLE_WIDTH)
                logging.debug(_pid_name[os.getpid()] + "    最后 %s 秒的音量: %s (buffer_len:+%s=%s text:'%s')" % (RMS_LAST_TIME, volume,len(data['audio']), lb, info["text"][:15]))
                if volume <= RMS_HOLDER:
                    logging.debug(_pid_name[os.getpid()] + "    最后 %s 秒的音量小于阈值, 清空buffer" % RMS_LAST_TIME)
                    logging.debug(_pid_name[os.getpid()] + "    清空前再转录一次文本")
                    text = model.transcribe_buffer(info["buffer"],
                                                   sr_inp=SAMPLE_RATE,
                                                   channels_inp=CHANNELS,
                                                   language=lang,
                                                   fp16=False)
                    eos_tag = True
                    info["buffer"] = b""
                    # 在锁外面发送消息x
                    # 在锁内发消息吧
                    if eos_tag:
                        logging.debug(_pid_name[os.getpid()] + "    识别到结束，发送最终文本 '%s...'" % text[:20])
                        rsp = json.dumps({"text": text, "mid": "0", "trace_id": data.get("trace_id","")})
                        # socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
                        q_output.put((rsp, sid))
                        _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_speech2text" % PORT)

                sid_info.update({sid: info})

        logging.debug(_pid_name[os.getpid()] + "    with lock结束")

        text = ""
        if len(info["buffer"]) > 0:
            logging.debug(_pid_name[os.getpid()] + "    buffer非空，开始转录")
            text = model.transcribe_buffer(info["buffer"],
                                           sr_inp=SAMPLE_RATE,
                                           channels_inp=CHANNELS,
                                           language=lang,
                                           fp16=False)
            logging.debug(_pid_name[os.getpid()] + "    transcribed: '%s'" % text)
            rsp = json.dumps({"text": text, "mid": "1", "trace_id": data.get("trace_id","")})

            # queue.task_done()
            logging.debug(_pid_name[os.getpid()] + "    Process of sid-%s-%s finished.(elapsed %.4f)" % (sid, t_str, time.time() - t_begin))
            # socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
            q_output.put((rsp, sid))
            _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_speech2text" % PORT)
            logging.debug(_pid_name[os.getpid()] + "size of data_queue: %s" % q_input.qsize())
        else:
            logging.debug(_pid_name[os.getpid()] + "    buffer为空")

        with lock:
            if sid in sid_info:
                info = sid_info[sid]
                info["text"] = text

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
        logging.info("will send to clinet.")
        socketio.emit("get_server_info", {"data": "this is a test message"}, namespace=NAME_SPACE)
        socketio.emit("audio_rsp", {"text": "manually debug send."}, namespace=NAME_SPACE)
        if DEBUG:
            for sid, audio in AUDIO_RECORD.items():
                wf_sid = wave.open(os.path.join(OUTPUT_DIR, 'audio_record_%s.wav' % sid), 'wb')
                wf_sid.setnchannels(CHANNELS)
                wf_sid.setsampwidth(SAMPLE_WIDTH)
                wf_sid.setframerate(SAMPLE_RATE)
                wf_sid.writeframes(audio)

        with LOCK:
            with open(os.path.join(OUTPUT_DIR, "debug.pkl"), "wb") as fwb:
                pickle.dump(dict(SID_INFO), fwb)
            for sid in SID_INFO:
                wf_sid = wave.open(os.path.join(OUTPUT_DIR, 'sid_%s.wav' % sid), 'wb')
                wf_sid.setnchannels(CHANNELS)
                wf_sid.setsampwidth(SAMPLE_WIDTH)
                wf_sid.setframerate(SAMPLE_RATE)
                wf_sid.writeframes(SID_INFO[sid]["buffer"])
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

    @socketio.on("speech2text", namespace=NAME_SPACE)
    def speech2text(data):
        data = json.loads(data)
        data["audio"] = base64.b64decode(data["audio"])
        sid = request.sid
        # ts = int(time.time())
        ts = int(data['ts'])
        t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        logging.debug(f"{NAME_SPACE}/speech2text received an input.(%s %s)" % (ts, t_str))
        if DEBUG:
            AUDIO_RECORD.update({sid: AUDIO_RECORD.get(sid, b"") + data["audio"]})

        # 将数据添加到队列中
        Q_speech2text.put((data, t_str, sid))
        logging.debug("size(estimate) of Q_speech2text: %s" % Q_speech2text.qsize())

    @app.route("/exec_emit_speech2text")
    def exec_emit_speech2text():
        # logging.debug("触发了向客户端发送消息（持续从队列中加载直到没有可加载的）")
        while True:
            try:
                rsp, sid = Q_speech2text_rsp.get_nowait()
                # logging.debug("    队列拿到结果 —— rsp:'%s...'" % str(rsp[:15]))
                socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
            except Empty:
                # logging.debug("    队列为空，退出")
                break
        return ""

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
                socketio.emit("text2speech_rsp", rsp, to=sid, namespace=NAME_SPACE)
            except Empty:
                # logging.debug("    队列为空，退出")
                break
        return ""

    @socketio.on("upload_speaker", namespace=NAME_SPACE)
    def upload_speaker(data):
        print("upload_speaker received ...")
        data = json.loads(data)
        data["audio"] = base64.b64decode(data["audio"])
        ts = int(time.time())
        t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"{NAME_SPACE} upload_speaker received an input.(%s %s)" % (ts, t_str))

        wf_sid = wave.open(os.path.join(VOICE_SAMPLE_DIR, data["speaker"]+"_"+data["language"]+".wav"), 'wb')
        wf_sid.setnchannels(CHANNELS)
        wf_sid.setsampwidth(SAMPLE_WIDTH)
        wf_sid.setframerate(SAMPLE_RATE)
        wf_sid.writeframes(data['audio'])
        socketio.emit("upload_speaker_rsp", json.dumps({"status": "0"}), to=request.sid, namespace=NAME_SPACE)

    return app, socketio


if __name__ == '__main__':
    mp.set_start_method("forkserver")
    manager = mp.Manager()
    Q_speech2text = manager.Queue()
    Q_speech2text_rsp = manager.Queue()
    Q_text2speech = manager.Queue()
    Q_text2speech_rsp = manager.Queue()
    SID_INFO = manager.dict()
    LOCK = mp.Lock()
    _PID_NAME = manager.dict()
    _PID_NAME[os.getpid()] = "主进程"
    processes = []
    for idx in range(PROCESS_NUM):
        p1 = mp.Process(target=process_queue_speech2text,
                        args=(Q_speech2text, Q_speech2text_rsp, SID_INFO, LOCK, _PID_NAME))
        p1.start()
        processes.append(p1)
        _PID_NAME.update({p1.pid: "子进程%s" % idx})
        logging.info("    speech2text子进程启动 (%s)" % p1.pid)

        p2 = mp.Process(target=process_queue_text2speech,
                        args=(Q_text2speech, Q_text2speech_rsp))
        p2.start()
        processes.append(p2)
        _PID_NAME.update({p2.pid: "子进程%s" % idx})
        logging.info("    text2speech子进程启动 (%s)" % p2.pid)
        
    logging.info(">>> MultiProcess ready. all pid as follow:")
    logging.info(_PID_NAME)

    app, socketio = create_app()
    logging.info("\n"*2+">>> Socket App run..."+"\n"*2)
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False)  # 阻塞
    # gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 server_sounda_mp:app
