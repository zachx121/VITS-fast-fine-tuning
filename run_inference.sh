
cp ./configs/modified_finetune_speaker.json ./finetune_speaker.json
python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --share True
