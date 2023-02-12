import os
"""import data
import webbrowser

topics_lang_count = data.topics["language"].value_counts()
contents_lang_count = data.contents["language"].value_counts()

langs_to_translate = topics_lang_count.loc[topics_lang_count > 100].index.intersection(contents_lang_count.loc[contents_lang_count > 100].index)
print(langs_to_translate)
print(topics_lang_count.loc[topics_lang_count > 100].index)
print(contents_lang_count.loc[contents_lang_count > 100].index)

print()
print()
print("LANG - Translate Links")

for lang in langs_to_translate:
    if lang != "en":
        print("https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + lang +"-en")
        # webbrowser.open("https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + lang +"-en")

# download manually VIA github

# non-existing models:
# sw, gu, fil, my, km, kn
# zh - manually download here: https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/models/zho-eng/README.md

# extra models:
extra_langs = ["de", "fi", "hu", "id", "ja", "ko", "nl", "pl", "ru", "sv"]
for lang in extra_langs:
    if lang != "en":
        print("https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + lang +"-en")
        # webbrowser.open("https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/" + lang +"-en")
"""
# note that some models use BPE encoder instead of sentencepiece encoder. we list them out
bpe_langs = []
for lang_files in os.listdir("data/opus_translation_models"):
    lang = lang_files[:2]
    if lang_files.endswith(".zip") and not os.path.isfile("data/opus_translation_models/" + lang + "-en/source.spm"):
        bpe_langs.append(lang)
print(bpe_langs)

# we have to manually download from OPUS for BPE langs.
# bengali (bn): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/ben-eng
# spanish (es): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/spa-eng
# hungarian (hu): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/hun-eng
# portuguese (pt): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/por-eng
# russian (ru): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/rus-eng
# assamese (as): https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/inc-inc

# we have to manually download some from the challenge also:
# swahili (sw) https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/swa-eng
# gujarati (gu) https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/ine-eng