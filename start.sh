#git clone https://github.com/zachx121/VITS-fast-fine-tuning.git
#pip install -r requirements.txt
#sh dependencies.sh
#rm -rf models && ln -s /input0 models
#apt update && apt install -y portaudio19-dev python3-pyaudio ffmpeg cmake
#pip uninstall cmake  # pyopenjtalk 的安装会用cmake，但是汇报一些not found的错误，先删掉再让它自己装就行
#pip install -r requirements.txt --no-cache-dir
#pip install openjtalk~=0.3.0
#pip install --no-index --find-links=packages -r _requirements.txt
python server_sounda.py 8080 "./models/sounda_voice_models_v1" "./models"
#python server.py