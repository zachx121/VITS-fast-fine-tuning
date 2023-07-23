

# 先用english2ipa拿到国际注音的ipa，然后再把ipa注音转成romaji注音
def ipa2romaji():
    # [debug]
    import text
    import json
    from utils_audio import play_audio
    from text2speech import Text2Speech

    M_tts = Text2Speech(model_dir="./vits_models/G_latest_cxm_1st.pth",
                        config_fp="./configs/finetune_speaker.json").init()
    def pronounce(symbol_inp):
        sr, audio = M_tts._symbols2audio(symbol_inp)
        play_audio(audio.tobytes(), sr)

    text.mandarin.chinese_to_romaji("这是我的炸酱面")
    text.cleaners.zh_ja_mixture_cleaners("[ZH]这是我的炸酱面[ZH]")
    text.mandarin.chinese_to_romaji("深圳")
    txt = "Hello, My name is norris. Today is a good day."
    txt_ipa = text.english.english_to_ipa(txt)
    txt_lzyipa = text.english.english_to_lazy_ipa(txt)
    txt_rmj = text.english.english_to_romaji(txt)
    with open("/Users/didi/0-Code/VITS-fast-fine-tuning/configs/finetune_speaker.json", "r") as fr:
        conf = json.load(fr)
        symbols = "".join(conf["symbols"])

    print(txt)
    print(txt_ipa)
    print(txt_lzyipa)
    print(txt_rmj)
    print("".join([i if i in conf["symbols"] else "[" + i + "]" for i in txt_rmj]))
    #sr, audio = M_tts._symbols2audio(txt_rmj)
    pronounce(txt_rmj)
    # hɛˈloʊ, maɪ neɪm ɪz ˈnɔrɪs. təˈdeɪ ɪz ə gʊd deɪ.
    pronounce("hə`loɯ→, mai→ nei→m i→z `no→ɹi→s. tə`dei→ i→z ə guɯ→d dei→.")

    # təˈdeɪ ɪz ə gʊd deɪ
    print(text.english.english_to_ipa("Today is a good day"))
    print(text.english.english_to_lazy_ipa("Today is a good day"))
    # tə↓dei iz ə gɯd dei
    print(text.english.english_to_romaji("Today is a good day"))
    # _,.!?-~…AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑

    pronounce("tə_dei→ iz ə→ gu→d dei→.")
    pronounce("tə~dei→↑ is ə~ gu→d dei→.")
