import logging
import pickle
import random
import whisper.tokenizer
import utils_audio

logging.basicConfig(format='[%(asctime)s-%(thread)d-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
from speech2text import Speech2Text
#from text2speech import Text2Speech
from sounda_voice.text2speech import Text2Speech
from flask import Flask, render_template, request, current_app
from flask_socketio import SocketIO, emit,  join_room
from concurrent.futures import ThreadPoolExecutor
import time
import audioop
import torch
import sys
import threading
import base64
import json
import os
sys.path.append("./")

# [Params]
PORT = int(sys.argv[1]) if len(sys.argv) >= 2 else 8080
# TTS_MODEL = sys.argv[2] if len(sys.argv) >= 3 else "./vits_models/G_latest_cxm_1st.pth"
TTS_MODEL_DIR = sys.argv[2] if len(sys.argv) >= 3 else "./sounda_voice_model_v1"
SST_MODEL_DIR = sys.argv[3] if len(sys.argv) >= 4 else "./whisper_models"
OUTPUT_DIR = "./output"
VOICE_SAMPLE_DIR = "./voice_sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
logging.info(">>> [PORT]: %s" % PORT)
logging.info(">>> [TTS_MODEL_DIR]: %s" % TTS_MODEL_DIR)
logging.info(">>> [SST_MODEL_DIR]: %s" % SST_MODEL_DIR)


# [Model prepared]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(">>> Construct Model (device is '%s')" % DEVICE)
M_tts = Text2Speech(encoder_fp=os.path.join(TTS_MODEL_DIR, "encoder.pt"),
                    synth_fp=os.path.join(TTS_MODEL_DIR, "synth.pt"),
                    vocoder_fp=os.path.join(TTS_MODEL_DIR, "vocoder.pt"))
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
SID_INFO = {}
LOCK = threading.Lock()
SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS
RMS_HOLDER = 777

# 以「127.0.0.1:8080/debug」这个页面的访问，来触发一次服务端对客户端的socket消息发送
# curl 127.0.0.1:8080/debug
@app.route("/debug")
def debug_func():
    logging.info("will send to clinet.")
    socketio.emit("get_server_info", {"data": "this is a test message"}, namespace=NAME_SPACE)
    socketio.emit("audio_rsp", {"text": "manually debug send."}, namespace=NAME_SPACE)
    with open(os.path.join(OUTPUT_DIR, "/debug.pkl"), "wb") as fwb:
        pickle.dump(SID_INFO, fwb)
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
    with LOCK:
        SID_INFO.pop(request.sid)


from datetime import datetime
import queue
import threading
# 创建一个线程安全的队列
queue_speech2text = queue.Queue()
queue_text2speech = queue.Queue()
executor = ThreadPoolExecutor(max_workers=10)  # 10个工作线程
# 创建一个线程池
# executor = ThreadPoolExecutor(max_workers=5)

# [DEBUG] 打开一个wav文件，每次收到的buffer都往里面写
import wave

# 子线程用whisper处理语音转文本
def process_queue_speech2text():
    logging.info("process_queue_speech2text start.")
    while True:
        # 从队列中获取数据
        data, t_str, sid = queue_speech2text.get()
        ts_data = data['ts']
        lang = data.get("language", None)
        lang = lang if lang in whisper.tokenizer.LANGUAGES else None

        if data is None:
            logging.info("data is None, break now. %s" % t_str)
            break

        t_begin = time.time()
        logging.debug("Process of sid-%s-%s start." % (sid, t_str))

        eos_tag =False
        # 处理数据
        with LOCK:
            if sid in SID_INFO:
                info = SID_INFO[sid]
                info["buffer"] += data['audio']
                info["last_active_time"] = ts_data

                # 最后一秒的音量
                lb = len(info["buffer"])
                volume = audioop.rms(info["buffer"][lb - int(0.5 * BYTES_PER_SEC):lb], SAMPLE_WIDTH)
                logging.debug("    最后0.5秒的音量: %s" % volume)
                if volume <= RMS_HOLDER:
                    logging.debug("    最后0.5秒的音量小于阈值, 清空buffer")
                    eos_tag = True
                    info["buffer"] = b""
                SID_INFO.update({sid: info})

        # 在锁外面发送消息
        if eos_tag:
            rsp = json.dumps({"text": info["text"], "mid": "0"})
            socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)

        text = ""
        if len(info["buffer"]) > 0:
            text = M_stt.transcribe_buffer(info["buffer"],
                                           sr_inp=SAMPLE_RATE,
                                           channels_inp=CHANNELS,
                                           language=lang,
                                           fp16=False)
            logging.debug("    transcribed: '%s'" % text)
            rsp = json.dumps({"text": text, "mid": "1"})

            queue_speech2text.task_done()
            logging.debug("    Process of sid-%s-%s finished.(elapsed %.4f)" % (sid, t_str, time.time() - t_begin))
            socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
            logging.debug("size of data_queue: %s" % queue_speech2text.qsize())

        with LOCK:
            if sid in SID_INFO:
                info = SID_INFO[sid]
                info["text"] = text

# 子线程用vits进行声音合成（文本转语音）
def process_queue_text2speech():
    logging.info("process_queue_text2speech start.")
    while True:
        # 从队列中获取数据
        data, t_str, sid = queue_text2speech.get()
        ts_data = int(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").timestamp())

        if data is None:
            logging.info("data is None, break now. %s" % t_str)
            break

        t_begin = time.time()
        logging.debug("  Process of sid-%s-%s start." % (sid, t_str))
        mock_voice_fp = os.path.join(VOICE_SAMPLE_DIR, data["speaker"]+"_"+data["language"]+".wav")
        if os.path.exists(mock_voice_fp):
            # 处理数据
            audio, sr = M_tts.synth_file(texts=data["text"],
                                         mock_audio_fp=mock_voice_fp)
            rsp = {"audio_buffer": base64.b64encode(audio.tobytes()).decode(),
                   "sr": str(sr),
                   "status": "0",
                   "msg": "success."}
            rsp = json.dumps(rsp)
            queue_text2speech.task_done()
            logging.debug("  Process of sid-%s-%s finished.(elapsed %s)" % (sid, t_str, time.time() - t_begin))
            socketio.emit("text2speech_rsp", rsp, to=sid, namespace=NAME_SPACE)
            logging.debug("size of data_queue: %s" % queue_text2speech.qsize())
        else:
            logging.error("mock_voice_fp not found at '%s'" % mock_voice_fp)
            rsp = {"audio_buffer": "",
                   "sr": "",
                   "status": "1",
                   "msg": "fail. not found speaker '%s'" % data['speaker']}
            rsp = json.dumps(rsp)
            queue_text2speech.task_done()
            socketio.emit("text2speech_rsp", rsp, to=sid, namespace=NAME_SPACE)

# 创建并启动一个新线程来处理队列中的数据
# threading.Thread(target=process_queue, daemon=True).start()
for _ in range(5):
    socketio.start_background_task(target=process_queue_speech2text)
    socketio.start_background_task(target=process_queue_text2speech)

@socketio.on("speech2text", namespace=NAME_SPACE)
def speech2text(data):
    data = json.loads(data)
    data["audio"] = base64.b64decode(data["audio"])
    # ts = int(time.time())
    ts = int(data['ts'])
    t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    logging.debug(f"{NAME_SPACE}_audio received an input.(%s %s)" % (ts, t_str))

    # 将数据添加到队列中
    queue_speech2text.put((data, t_str, request.sid))
    logging.debug("    size(estimate) of data_queue: %s" % queue_speech2text.qsize())


@socketio.on("text2speech", namespace=NAME_SPACE)
def text2speech(data):
    # 放到子线程里做
    data = json.loads(data)
    ts = int(time.time())
    t_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{NAME_SPACE}_audio received an input.(%s %s)" % (ts, t_str))

    # 将数据添加到队列中
    queue_text2speech.put((data, t_str, request.sid))
    logging.debug("size(estimate) of data_queue: %s" % queue_speech2text.qsize())

    #
    # sr, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
    # rsp = {"text": text, "audio_buffer": audio.tobytes(), "sr": sr}
    # socketio.emit("audio_rsp", rsp, to=request.sid, namespace=NAME_SPACE)

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
