import pandas as pd
import numpy as np
import gc
import time
import os
import psutil
import sentence_transformers
import collections
import multiprocessing

import model_bert_preprocessing_pipeline

current_process = psutil.Process(os.getpid())


def print_memory():
    memory_percent = current_process.memory_percent()
    print("memory percent: " + str(memory_percent) + "%")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")


    print_memory()

    # TEXT CLEANING
    ctime = time.time()
    p = multiprocessing.Process(target=model_bert_preprocessing_pipeline.text_cleaning_subprocess)
    p.start()
    p.join()
    del p
    ctime = time.time() - ctime
    print()
    print("Preprocessing time used: ", ctime)

    print_memory()

    # TRANSLATION
    p = multiprocessing.Process(target=model_bert_preprocessing_pipeline.translate_subprocess)
    p.start()
    p.join()
    del p
    print_memory()

    data_topics = pd.read_csv("generated_data/new_topics.csv", index_col=0)
    data_contents = pd.read_csv("generated_data/new_contents.csv", index_col=0)
    print("After loading memory:")
    print_memory()

    languages = ['ar', 'as', 'bn', 'en', 'es', 'gu', 'hi', 'mr', 'pt',
                 'zh']  # languages belonging to the train data (in the public contents.csv, topics.csv)
    languages2 = ['ar', 'as', 'bg', 'bn', 'en', 'es', 'fr', 'gu', 'hi', 'it', 'mr', 'pl', 'pt', 'ru', 'sw',
                  'zh']  # total data

    topics_lang_one_hot = pd.DataFrame(
        {lang: (data_topics["language"] == lang).astype(dtype=float) for lang in languages})
    topics_lang_one_hot2 = pd.DataFrame(
        {lang: (data_topics["language"] == lang).astype(dtype=float) for lang in languages2})

    contents_lang_one_hot = pd.DataFrame(
        {lang: (data_contents["language"] == lang).astype(dtype=float) for lang in languages})
    contents_lang_one_hot2 = pd.DataFrame(
        {lang: (data_contents["language"] == lang).astype(dtype=float) for lang in languages2})

    np.save("generated_data/topics_lang_train.npy", topics_lang_one_hot.to_numpy())
    np.save("generated_data/topics_lang_total.npy", topics_lang_one_hot2.to_numpy())
    np.save("generated_data/contents_lang_train.npy", contents_lang_one_hot.to_numpy())
    np.save("generated_data/contents_lang_total.npy", contents_lang_one_hot2.to_numpy())

    del topics_lang_one_hot, topics_lang_one_hot2, contents_lang_one_hot, contents_lang_one_hot2

    print_memory()

    # TEXT TO VECTORS
    def transform_text(x):
        if type(x) == str:
            x = x.replace("\n", " ").replace("_", " ").replace("-", " ").strip()
            x = x.replace("  ", " ").replace("  ", " ")
            tokens = x.split()
            freqs = dict(collections.Counter(tokens))
            for tok in freqs:
                if len(tok) > 0 and freqs[tok] > 6:
                    x = x.replace(" " + tok + " " + tok, " " + tok).replace(" " + tok + " " + tok, " " + tok)
            return x.strip()
        return x


    def remove_problematic_titles(x):
        if type(x) == str:
            if len(x.split()) > 70:
                return np.nan
            return x
        return x


    minilm_L12_eng = sentence_transformers.SentenceTransformer(
        "transformer_models/all_minilm_l12_v2")
    mpnet_english_finetuned_on_train = sentence_transformers.SentenceTransformer(
        "transformer_models/mpnet_english_finetuned_on_train")

    def sbert_direct_vectorize_and_save_translate(series, name):
        has_content_mask = (~(series.isnull() | series.apply(lambda x: type(x) == str and x == ""))).to_numpy()
        series = series.fillna("")
        dirs = ["generated_data/mininet_L6_english384/", "generated_data/mininet_L12_english384/"]
        for mdir in dirs:
            if not os.path.isdir(mdir):
                os.mkdir(mdir)

        size = len(series)

        sentences = list(series)

        ctime = time.time()
        mininet384_L12 = minilm_L12_eng.encode(sentences)
        ctime = time.time() - ctime
        print("Time elapsed: ", ctime, "  for ", name, " mininet384_L12")

        np.save("generated_data/mininet_L12_english384/" + name + ".npy", mininet384_L12)
        del mininet384_L12, series, sentences

        return has_content_mask


    def sbert_combined_vectorize_and_save_translate(title_series, description_series, name):
        text_list = np.empty(shape=len(title_series), dtype="object")
        for k in range(len(title_series)):
            mtitle = title_series.iloc[k]
            mdesc = description_series.iloc[k]
            if type(mtitle) == float and type(mdesc) == float:
                mstr = ""
            elif type(mtitle) == float:
                mstr = mdesc
            elif type(mdesc) == float:
                mstr = mtitle
            else:
                mstr = mtitle + " " + mdesc
            mtokens = mstr.split()
            if len(mtokens) > 50:
                mstr = " ".join(mtokens[:50])
            text_list[k] = mstr

        dirs = ["generated_data/mpnet_english_finetuned_on_train/"]
        for mdir in dirs:
            if not os.path.isdir(mdir):
                os.mkdir(mdir)

        size = len(title_series)
        sentences = list(text_list)

        ctime = time.time()
        finevectors = mpnet_english_finetuned_on_train.encode(sentences)
        ctime = time.time() - ctime
        print("Time elapsed: ", ctime, "  for ", name, " mpnet_english_finetuned_on_train")
        np.save("generated_data/mpnet_english_finetuned_on_train/" + name + ".npy", finevectors)
        del finevectors, sentences, text_list


    print("Start full pipeline vectorizing text (sbert, translated)")
    ctime = time.time()
    contents_description_mask = sbert_direct_vectorize_and_save_translate(data_contents["description_translate"],
                                                                          "contents_description")
    print("Finished contents description")
    contents_title_mask = sbert_direct_vectorize_and_save_translate(data_contents["title_translate"], "contents_title")
    print("Finished contents title")
    topics_description_mask = sbert_direct_vectorize_and_save_translate(data_topics["description_translate"],
                                                                        "topics_description")
    print("Finished topics description")
    topics_title_mask = sbert_direct_vectorize_and_save_translate(data_topics["title_translate"], "topics_title")
    print("Finished topics title")
    ctime = time.time() - ctime
    print(ctime)

    print("Start full pipeline vectorizing text (sbert finetune, translated)")
    ctime = time.time()
    sbert_combined_vectorize_and_save_translate(data_contents["title_translate"],
                                                data_contents["description_translate"], "contents")
    print("Finished contents")
    sbert_combined_vectorize_and_save_translate(data_topics["title_translate"], data_topics["description_translate"],
                                                "topics")
    print("Finished topics")
    ctime = time.time() - ctime
    print(ctime)

    gc.collect()

    print_memory()