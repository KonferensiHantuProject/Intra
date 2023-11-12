import os

import urllib.request

def Helfiler():

    if os.path.isdir("document") == False:
        os.mkdir("document")

    tab_url = "https://github.com/sintiasnn/Itepee/raw/main/ItepeeApp/data/"

    tables = ["new_kamusalay.csv","spesial_characters_HTML.csv","stopwordbahasa.csv"]

    for t in tables:
        t_origin = tab_url + t
        urllib.request.urlretrieve(t_origin, f"document/{t}")

def Helmodel():

    if os.path.isdir("model") == False:
        os.mkdir("model")

    mod_url = "https://github.com/sintiasnn/Itepee/raw/main/ItepeeApp/model/"

    model = [
        "model1_fold_batch32_epoch20_lr0.0001.h5",
        "model2_fold_batch32_epoch20_lr0.0001.h5",
        "model3_fold_batch64_epoch20_lr0.001.h5",
        "model4_fold_batch32_epoch10_lr0.001.h5",
        "tokenizer_1.pickle"
    ]

    for m in model:
        m_origin = mod_url + m
        urllib.request.urlretrieve(m_origin, f"model/{m}")
