import ctranslate2
import sentencepiece as spm
import os
import config_translation
import langdetect
import numpy as np
import pandas as pd

langdetect.DetectorFactory.seed = 0


def get_language_model_folder(lang):
    return config_translation.resources_path + lang + "-en/"


def list_available_langs():
    return [lang_folder[:2] for lang_folder in os.listdir(config_translation.resources_path)]


def load_model(lang):
    lang_dir = get_language_model_folder(lang)
    if not os.path.isfile(lang_dir + "source.spm"):
        raise Exception("Language does not exist")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(lang_dir + "source.spm")
    return tokenizer, ctranslate2.Translator(lang_dir, device = "cuda")


def translate_sentences(sentences, translate_info):
    tokenizer, translator = translate_info
    tokens = tokenizer.encode(sentences, out_type=str)
    results = translator.translate_batch(tokens)
    output = tokenizer.decode([result.hypotheses[0] for result in results])
    output = [text.replace("▁", " ").replace("⁇ cmn_Hans ⁇", "").strip() for text in output]
    return output

def detect_language(text):
    try:
        langdetect.DetectorFactory.seed = 0
        probs = langdetect.detect_langs(text)
        probs = {probs[0].lang: probs[0].prob for x in probs}  # convert to dict
        """for lang in probs.keys():
            if lang.startswith("zh"):
                prob = probs[lang]
                probs.pop(lang, None)
                if "zh" in probs:
                    probs["zh"] = probs["zh"] + prob
                probs["zh"] = """  # note chinese is splitted into zh-tw, zh-cn. we do not need this yet since we use en only.
        return "en" in probs and probs["en"] > 0.9
    except:
        return False


# returns a list of bools, indicating whether the string is English
def detect_languages(sentences):
    lang_en = [detect_language(text) for text in sentences]
    return lang_en

def obtain_lang_subroutine(text, lang):
    if type(text) == str:
        if (" " not in text) and (sum(c.isdigit() for c in text) > 7) and (sum(c.isalpha() for c in text) > 7) and lang == "ar":
            return "none_ar"
        if (" " not in text) and ("source_id=" in text):
            return "none_src"
        if (" " not in text) and (len(text) < 6) and ("v" in text) and ("." in text) and (sum(c.isdigit() for c in text) > 1):
            return "none_small"
        if detect_language(text):
            return "en"
    return lang

def obtain_actual_languages(dframe, k):
    row = dframe.iloc[k]
    title = row["title"]
    description = row["description"]
    language = row["language"]
    if language == "en":
        return language, language
    title = obtain_lang_subroutine(title, language)
    description = obtain_lang_subroutine(description, language)
    return description, title


def obtain_language_info(dframe):
    description_lang = np.zeros(shape=(len(dframe)), dtype="object")
    title_lang = np.zeros(shape=(len(dframe)), dtype="object")
    for k in range(len(dframe)):
        description, title = obtain_actual_languages(dframe, k)
        description_lang[k] = description
        title_lang[k] = title
    return pd.DataFrame(data={"description_lang": description_lang, "title_lang": title_lang}, index=dframe.index)