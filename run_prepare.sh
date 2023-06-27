
export PATH="/nfs/project/opt/miniconda3/bin::$PATH"
conda config --add envs_dirs /nfs/project/opt/miniconda3/envs
source activate py38_torch2
#
#
#function mkSurePathExist() {
#    if [[ ! -d $1 ]]; then
#        mkdir $1
#    fi
#}
#mkSurePathExist raw_audio
#mkSurePathExist denoised_audio
#mkSurePathExist custom_character_voice
#mkSurePathExist segmented_character_voice
#mkSurePathExist pretrained_models
#
## 依赖项
#pip install -r requirements.txt
## 辅助训练语音和标注
#wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
#unzip sampled_audio4ft_v2.zip
## 编译monotonic
#cd monotonic_align/ && mkdir monotonic_align && python setup.py build_ext --inplace && cd ..
## 预训练模型
#wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth -O ./pretrained_models/D_0.pth
#wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth -O ./pretrained_models/G_0.pth
#wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/config.json -O ./configs/finetune_speaker.json
