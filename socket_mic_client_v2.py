import time

import socketio
import pyaudio
import audioop
import threading
import logging

logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

GEN_AUDIO = False
# 创建一个Socket.IO client
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


@sio.on("audio_rsp", namespace="/MY_SPACE")
def rsp_func(message):
    logging.info("[words]: %s" % message['text'])
    if GEN_AUDIO:
        sr, audio_buffer = message['sr'], message['audio_buffer']
        t = threading.Thread(target=play_audio, args=(audio_buffer, sr))
        t.start()
        t.join()

host = "http://localhost:8080"
# host = "https://zach-sgio6wfd7vze.serv-c1.openbayes.net"
sio.connect(host+'/MY_SPACE')
time.sleep(5)
sio.emit('my_event', {'data': 'Hello, World!'}, namespace='/MY_SPACE')


# 以下是PyAudio参数，可能需要根据你的麦克风和系统进行修改
chunk = 1024*10  # 每次读取的音频数据的长度
sample_format = pyaudio.paInt16  # 16位深度
sample_width = 2  # 标准的16位PCM音频中，每个样本占用2个字节
sample_rate = 44100  # 采样频率
channels = 1  # 音频通道数


p = pyaudio.PyAudio()

logging.info(">>> opening audio stream...")
stream = p.open(format=sample_format,
                channels=channels,
                rate=sample_rate,
                frames_per_buffer=chunk,
                input=True)

logging.info(">>> 开启录音，等待服务器就绪...")
sio.emit('init', {}, namespace='/MY_SPACE')
# 音量开始超过阈值时，持续添加音频数据到缓冲区，直到持续两秒小于阈值时停止添加，并在子进程将缓冲区的内容发送给服务器；
# 同时，还需要在子进程持续检测缓冲区的音频长度是否超过5秒，如果超过5秒直接发送给服务器并清空缓冲区；
# 在整个过程中，需要保证不阻塞主进程，以便接收到用户的语音输入。

# 缓冲区最长时间
buffer_max_sec = 15
# 最短停顿检测时间
gap_duration_holder = 0.5
# 音量阈值
threshold = 800  # 500会录到敲键盘的声音
# 缓冲区
buffer_cache = b""
# 音频间隔开始时间
gap_start = 0
# 发送音频
should_send = False
# 记录音频
should_record = False
# 音频字节流中每秒的字节大小
bytes_per_second = sample_rate * sample_width * channels
# 锁，用于同步访问 buffer_cache
lock = threading.Lock()


def send_data():
    global buffer_cache, should_send
    while True:
        # 缓冲区的音频时长超过5秒，发送数据并清空缓冲区
        buffer_length_seconds = len(buffer_cache) / bytes_per_second
        if buffer_length_seconds >= buffer_max_sec:
            logging.info("[SubProcess] Talking for more than %s second." % buffer_max_sec)
            should_send = True

        with lock:
            if should_send:
                logging.debug("[SubProcess] sending... (size is %s, sid is %s)"
                              % (len(buffer_cache), sio.get_sid("/MY_SPACE")))
                audio_info = {"audio": buffer_cache,
                              "channels": channels,
                              "sample_rate": sample_rate,
                              "gen_audio": GEN_AUDIO}
                sio.emit('process', audio_info, namespace='/MY_SPACE')
                buffer_cache = b""
                should_send = False


# 启动发送数据的线程
send_thread = threading.Thread(target=send_data)
send_thread.start()


try:

    while True:
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)  # 使用audioop.rms()函数计算音量

        if rms > threshold:
            # 大于阈值时，重置gap_start为0
            gap_start = 0
            should_record = True
        elif should_record:
            # 记录音频过程中出现音量低于阈值的情况
            if gap_start == 0:
                # 首次出现音量低于阈值，记录时间戳
                gap_start = time.time()
            elif time.time() - gap_start > gap_duration_holder:
                # 音量低于阈值已经持续2秒，停止记录，并发送数据
                logging.debug("Quiet for more than %s second." % gap_duration_holder)
                with lock:
                    should_record = False
                    should_send = True
        if should_record:
            logging.debug("   recording rms: %s, buffer_size: %s" % (rms, len(buffer_cache)))
            with lock:
                buffer_cache += data

except KeyboardInterrupt:
    print('停止录音')
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('断开连接')
    sio.disconnect()

