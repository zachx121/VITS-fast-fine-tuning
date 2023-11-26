
import json
import logging
import os

import requests
import time

mode = "speech2text"  # speech2text, text2speech
headers = {
    "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjIxMjM5MiwidXVpZCI6ImVmY2MyMWUwLWUwYzYtNDY4Ny05N2MwLTg4OTljYTQzYjRlZiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.RvIaNq0u2f0X-_gmhJgutOq4xY01XLAeXyhRJyrDH-1bUQwtLyIrgsvxvtw0yTAbnLE4oRk-jju1RkSIMmYsTg",
    "Content-Type": "application/json"
}
host = "https://api.autodl.com"

deploy = False
check = True

# 创建ReplicaSet类型部署
# 参数说明 https://www.autodl.com/docs/esd_api_doc/#_5
url = host+"/api/v1/dev/deployment"
body = {
    "name": "api自动创建",
    "deployment_type": "ReplicaSet",
    "replica_num": 2,
    "container_template": {
        "region_sign": "beijingDC1",  # 容器可调度的地区
        "gpu_name_set": ["RTX 3090"],  # 可调度的GPU型号
        "gpu_num": 1,
        "cuda_v": 113,  # 将选择GPU驱动支持该CUDA版本的主机进行调度
        "cpu_num_from": 6,
        "cpu_num_to": 100,
        "memory_size_from": 32,  # 可调度的容器内存大小范围。单位：GB
        "memory_size_to": 256,
        "cmd": "cd Serving_VITS-fast-fine-tuning/ && python server_vits_mp_speech2text.py > nohup.out 2>&1 && sleep infinity ",  # 启动容器命令
        "price_from": int(0.1*1000),  # 可调度的价格范围。单位：元 * 1000，如0.1元填写100
        "price_to": int(2*1000),
        "image_uuid": "image-8e6a5a391c",
    },
}
response = requests.post(url, json=body, headers=headers)
print(response.content.decode())
info = json.loads(response.content.decode())
deploy_uuid = info["deployment_uuid"]


# 轮训检查部署是否完成
for i in range(10):
    response = requests.post(host + "/api/v1/dev/deployment/list",
                             json={"page_index": 1, "page_size": 100},
                             headers=headers)
    deploy_rsp = json.loads(response.content.decode())
    deploy_info = [i for i in deploy_rsp['data']['list'] if i['uuid'] == deploy_uuid][0]
    if deploy_info["status"] != "running" or deploy_info["replica_num"] != deploy_info["running_num"]:
        logging.info("部署还未完成 ...")
    time.sleep(10)  # 10s


time.sleep(30)
# 拿到这次部署的所有container的service_url测试服务
body = {
    "deployment_uuid": deploy_uuid,
    "page_index": 1,
    "page_size": 200
}
response = requests.post(host + "/api/v1/dev/deployment/container/list", json=body, headers=headers)
container_info = json.loads(response.content.decode())
urls = [i["info"]["service_url"] for i in container_info["data"]["list"]]
print(urls)

for host in urls:
    os.system("python debug.py speech2text %s" % host)
