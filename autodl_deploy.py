
import json
import logging
import os
import sys
import requests
import time

assert len(sys.argv) >=3, "第一个参数是nums控制启动几个服务，第二个参数是mode控制speech2text/text2speech"
nums = sys.argv[1]  # e.g. 5表示启动5个docker
mode = sys.argv[2]  # speech2text, text2speech
# large模型在3090-24G上只能启动2个(21G); medium在v100-32G能启动4个(28G); base在v100-32G能启动17个(28G)
# base~1.7G; medium~7G; large~10.5G
# beam_size从5降到3，best_of从5降到1可以降低显存压力： medium~5.75G; large~10.5G
# type_speech2text, nums_speech2text = "large-v2", 2
type_speech2text, nums_speech2text = "medium", 3
# type_speech2text, nums_speech2text = "base", 12
lang, nums_text2speech = "en", 12
# en speakers: en_m_apple,en_m_armstrong,en_m_pengu,en_m_senapi,en_wm_Beth,en_wm_Boer,en_wm_Kathy,zh_m_AK,zh_m_daniel,zh_m_silang,zh_m_TaiWanKang,zh_wm_TaiWanYu,zh_wm_Annie
# zh speakers: 四郎配音,bruce,daniel,zhongli
image_uuid = "image-0067222da6"  # 服务部署12.03
cmd_dict = {
    "speech2text": f"cd Serving_VITS-fast-fine-tuning/ && python server_vits_mp_speech2text.py {nums_speech2text} {type_speech2text}",
    "text2speech": f"cd Serving_VITS-fast-fine-tuning/ && python server_vits_mp_text2speech.py {nums_text2speech} {lang}"
}

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
    "replica_num": int(nums),
    "container_template": {
        "region_sign": "beijingDC1",  # 容器可调度的地区
        "gpu_name_set": ["RTX 3090"],  # 可调度的GPU型号
        "gpu_num": 1,
        "cuda_v": 113,  # 将选择GPU驱动支持该CUDA版本的主机进行调度
        "cpu_num_from": 6,
        "cpu_num_to": 100,
        "memory_size_from": 32,  # 可调度的容器内存大小范围。单位：GB
        "memory_size_to": 256,
        "cmd": "%s > nohup.out 2>&1 && sleep infinity " % cmd_dict[mode],  # 启动容器命令
        "price_from": int(0.1*1000),  # 可调度的价格范围。单位：元 * 1000，如0.1元填写100
        "price_to": int(2*1000),
        "image_uuid": image_uuid,
    },
}
response = requests.post(url, json=body, headers=headers)
print(response.content.decode())
info = json.loads(response.content.decode())
deploy_uuid = info["data"]["deployment_uuid"]


print(">>> 轮训检查部署是否完成")
max_cnt=50
for i in range(max_cnt):
    response = requests.post(host + "/api/v1/dev/deployment/list",
                             json={"page_index": 1, "page_size": 100},
                             headers=headers)
    deploy_rsp = json.loads(response.content.decode())
    deploy_info = [i for i in deploy_rsp['data']['list'] if i['uuid'] == deploy_uuid][0]
    if deploy_info["status"] != "running" or deploy_info["replica_num"] != deploy_info["running_num"]:
        print(f"部署还未完成 (status:{deploy_info['status']}, starting:{deploy_info['starting_num']}, running:{deploy_info['running_num']})")
        time.sleep(5)  # 10s
        print(f"镜像启动完成 (status:{deploy_info['status']}, starting:{deploy_info['starting_num']}, running:{deploy_info['running_num']})")
        print("sleep 10s等待镜像内服务初始化")
        break


time.sleep(10)
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

# for host in urls:
#     os.system("python debug.py %s %s" % (mode, host))
