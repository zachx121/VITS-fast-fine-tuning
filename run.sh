
export PATH="/nfs/project/opt/miniconda3/bin::$PATH"
conda config --add envs_dirs /nfs/project/opt/miniconda3/envs
source activate py38_torch2

export LD_LIBRARY_PATH=/nfs/project/opt/miniconda3/envs/py38_torch2/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
# 对所有上传的数据进行自动去背景音&标注, 需要调用Whisper和Demucs，运行时间可能较长。
# 将所有音频（无论是上传的还是从视频抽取的，必须是.wav格式）去噪
python scripts/denoise_audio.py
# 分割并标注长音频
python scripts/long_audio_transcribe.py --languages "CJ" --whisper_size large
# 标注短音频
python scripts/short_audio_transcribe.py --languages "CJ" --whisper_size large
# 底模采样率可能与辅助数据不同，需要重采样
python scripts/resample.py

# 选择是否加入辅助训练数据：/ Choose whether to add auxiliary data:
# ADD_AUXILIARY=True
python preprocess_v2.py --add_auxiliary_data True --languages "CJ"
# ADD_AUXILIARY=False
# python preprocess_v2.py --languages "CJ"


#import os
#os.environ['TENSORBOARD_BINARY'] = '/usr/local/bin/tensorboard'
#%reload_ext tensorboard
#%tensorboard --logdir "./OUTPUT_MODEL"
python finetune_speaker_v2.py -m "./OUTPUT_MODEL" --max_epochs "100" --drop_speaker_embed True
