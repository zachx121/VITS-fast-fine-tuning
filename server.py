import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from flask import Flask, render_template
from flask_socketio import SocketIO, emit


from speech2text import Speech2Text
from text2speech import Text2Speech
M_stt = Speech2Text(model_type="medium", download_root="./whisper_models").init()
M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                    config_fp="./configs/finetune_speaker.json").init()



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*')
name_space = '/MY_SPACE'


def socket_accept_audio(data):
    #todo. 接受客户端的音频数据buffer
    some_inp = data['audio']
    ###
    text = M_stt.transcribe_data(some_inp)
    audio = M_tts.gen_audio()
    sample_rate, audio = M_tts.tts_fn(text, speaker="audio", language="简体中文", speed=1.0)
    ###
    #todo. 把采样率和audio返回给客户端
    socketio.emit()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=6666, debug=True)
