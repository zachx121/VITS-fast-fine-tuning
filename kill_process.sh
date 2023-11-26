ps -ef | grep python | grep multiprocessing | awk '{print $2}' | xargs kill
ps -ef | grep python | grep server_vits | awk '{print $2}' | xargs kill
