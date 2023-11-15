ps -ef | grep python | grep multiprocessing | awk '{print $2}' | xargs kill
ps -ef | grep python | grep server_vits_mp_tts.py | awk '{print $2}' | xargs kill
