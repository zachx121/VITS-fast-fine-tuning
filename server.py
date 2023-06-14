import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s-SERVER]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from speech2text import Speech2Text
from text2speech import Text2Speech
from flask import Flask, render_template
from flask_socketio import SocketIO, emit,  join_room
import numpy as np

# [Flask Service init]
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*')
NAME_SPACE = '/MY_SPACE'

logging.info(">>> Construct Model")
M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                    config_fp="./configs/finetune_speaker.json")
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


@socketio.on("process", namespace=NAME_SPACE)
def process_audio(data):
    logging.info(f"{NAME_SPACE}_audio received an input.")
    text = M_stt.transcribe_buffer(data['audio'],
                                   sr_inp=data['sample_rate'],
                                   channels_inp=data['channels'],
                                   fp16=False)

    rsp = {"data": text}
    emit("audio_rsp", rsp)
    logging.info("send message to client: %s" % rsp)


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
