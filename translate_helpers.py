import ctranslate2
import sentencepiece as spm
import os
import config_translation
import numpy as np
import pandas as pd
import langid

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
    output = [text.replace("▁", " ").replace("⁇ cmn_Hans ⁇", "").replace("⁇ cmn_Hant ⁇", "").strip() for text in output]
    return output

lang_detect_model = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)
def detect_language(text):
    try:
        langid.classify("")
        result = lang_detect_model.classify(text)
        prob = result[1]
        lang = result[0]
        return lang == "en" and prob > 0.75
    except:
        return False


# returns a list of bools, indicating whether the string is English
def detect_languages(sentences):
    lang_en = [detect_language(text) for text in sentences]
    return lang_en

def obtain_lang_subroutine(text, lang):
    if type(text) == str:
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