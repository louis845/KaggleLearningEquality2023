def text_cleaning_subprocess():
    import langid
    import pandas as pd
    import numpy as np
    import os

    data_contents = pd.read_csv("data/content.csv", index_col=0)
    data_topics = pd.read_csv("data/topics.csv", index_col=0)

    lang_detect_model = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)

    # scraped from https://fileinfo.com/filetypes/common
    common_files_arr = np.array(
        [".doc", ".docx", ".eml", ".log", ".msg", ".odt", ".pages", ".rtf", ".tex", ".txt", ".wpd", ".aae", ".bin",
         ".csv", ".dat", ".key", ".mpp", ".obb", ".ppt", ".pptx", ".rpt", ".sdf", ".tar", ".vcf", ".xml", ".aif",
         ".flac", ".m3u", ".m4a", ".mid", ".mp3", ".ogg", ".wav", ".wma", ".3gp", ".asf", ".avi", ".flv", ".m4v",
         ".mov", ".mp4", ".mpg", ".srt", ".swf", ".ts", ".vob", ".wmv", ".3dm", ".3ds", ".blend", ".dae", ".fbx",
         ".max", ".obj", ".bmp", ".dcm", ".dds", ".djvu", ".gif", ".heic", ".jpg", ".png", ".psd", ".tga", ".tif",
         ".ai", ".cdr", ".emf", ".eps", ".ps", ".sketch", ".svg", ".vsdx", ".afpub", ".indd", ".oxps", ".pdf", ".pmd",
         ".pub", ".qxp", ".xps", ".numbers", ".ods", ".xlr", ".xls", ".xlsx", ".accdb", ".crypt14", ".db", ".mdb",
         ".odb", ".pdb", ".sql", ".sqlite", ".apk", ".app", ".bat", ".bin", ".cmd", ".com", ".exe", ".ipa", ".jar",
         ".run", ".sh", ".bin", ".dem", ".gam", ".gba", ".nes", ".pak", ".pkg", ".rom", ".sav", ".sav", ".dgn", ".dwg",
         ".dxf", ".step", ".stl", ".stp", ".gpx", ".kml", ".kmz", ".osm", ".asp", ".aspx", ".cer", ".cfm", ".csr",
         ".css", ".html", ".js", ".json", ".jsp", ".php", ".xhtml", ".crx", ".ecf", ".plugin", ".safariextz", ".xpi",
         ".fnt", ".otf", ".ttf", ".woff", ".woff2", ".ani", ".cab", ".cpl", ".cur", ".deskthemepack", ".dll", ".dmp",
         ".drv", ".icns", ".ico", ".lnk", ".reg", ".sys", ".cfg", ".ini", ".pkg", ".prf", ".set", ".asc", ".bin",
         ".enc", ".mim", ".uue", ".7z", ".cbr", ".deb", ".gz", ".pak", ".pkg", ".rar", ".rpm", ".tar.gz", ".xapk",
         ".zip", ".zipx", ".bin", ".dmg", ".img", ".iso", ".mdf", ".rom", ".vcd", ".appx", ".c", ".class", ".config",
         ".cpp", ".cs", ".h", ".java", ".kt", ".lua", ".m", ".md", ".pl", ".py", ".sb3", ".sln", ".swift", ".unity",
         ".vb", ".vcxproj", ".xcodeproj", ".yml", ".abk", ".arc", ".bak", ".tmp", ".crdownload", ".ics", ".msi",
         ".nomedia", ".part", ".pkpass", ".torrent"], dtype="object")
    common_files_arr.sort()

    def is_token_code(text, lang):
        if (lang == "zh" or lang == "th" or lang == "lo" or lang == "km" or lang == "bo"
                or lang == "ko" or lang == "ja" or lang == "jv" or lang == "my"):
            return False
        return (" " not in text) and (sum(c.isdigit() for c in text) > 7) and (sum(c.isalpha() for c in text) > 7)

    def is_token_source(text):
        return (" " not in text) and ("source_id=" in text)

    def is_token_version(text):
        subtext = text[1:]
        digits = sum(c.isdigit() for c in subtext)
        return text[0] == "v" and ((digits > 2) or (digits > 1 and "." in subtext))

    def is_token_bad_token(text, lang):
        if len(text) < 15 or lang == "zh":
            return False
        lang_res = lang_detect_model.classify(text)
        detect_lang = lang_res[0]
        prob = lang_res[1]
        return not (((detect_lang == lang) and prob > 0.75) or ((detect_lang == "en") and prob > 0.75))

    def is_token_css(text):
        return (text.startswith(".") or text.startswith("{") or text.endswith("}")) and "-" in text and (
                    text.endswith(":") or text.endswith("}") or text.endswith("{"))

    def is_token_latex(text):
        return (text.startswith("\\") and ("{" in text or "}" in text or "_" in text or "^" in text)) or (
                    "{" in text and "}" in text and "\\" in text)

    def manage_token_website(text):
        if "http://" in text or "https://" in text or "www." in text:
            if text.endswith(")") or text.endswith("]"):
                text = text[:-1]

            if text.endswith("/"):
                text = text[:-1]
            pos = text.rfind("/")
            if pos == -1:
                return []
            url_last = text[(pos + 1):]
            if len(url_last) == 0:
                return []
            url_last = url_last.replace("_", " ").replace("-", " ")
            url_toks = url_last.split(" ")

            # if it has a .something suffix and is in the set of common file types, remove it.
            dot_loc = url_toks[-1].rfind(".")
            if dot_loc != -1:
                if np.searchsorted(common_files_arr, url_toks[-1][dot_loc:], side="right") > np.searchsorted(
                        common_files_arr, url_toks[-1][dot_loc:], side="left"):
                    url_toks[-1] = url_toks[-1][:dot_loc]
            return ["( web resource:"] + url_toks + [")"]
        return [text]

    def manage_token_file(text):
        potfiles = text.split(".")

        potlist = np.array([("." + dottext.lower()) for dottext in potfiles], dtype="object")
        contains_pos = np.searchsorted(common_files_arr, potlist, side="right") > np.searchsorted(common_files_arr,
                                                                                                  potlist, side="left")
        contains_pos[0] = False
        if contains_pos.astype(np.int32).sum() > 0:
            not_contains_pos = np.logical_not(contains_pos)
            # filter out all the file suffixes
            return "( file resource: " + (
                " ".join(list(np.array(potfiles, dtype="object")[not_contains_pos])).replace("-", " ").replace("_",
                                                                                                               " ")) + " )"
        return text

    def transform_websites_and_files(text):
        tokens = text.split(" ")
        web_file_transform = [manage_token_file(tok2) for tok in tokens for tok2 in manage_token_website(tok)]
        web_file_transform = [tok2 for tok in web_file_transform for tok2 in tok.split()]
        return web_file_transform

    def remove_unwanted_tokens(tltuple):
        text, lang = tltuple
        if type(text) != str:
            return text
        # first reduce websites and files info, and then remove unwanted tokens
        good_tokens = [tok for tok in transform_websites_and_files(text) if not
        (is_token_code(tok, lang) or is_token_source(tok) or is_token_version(tok))]
        if len(good_tokens) == 0:
            return np.nan
        result = " ".join(good_tokens).replace("( file resource: )", "").replace("( web resource: )", "").replace("_",
                                                                                                                  " ").replace(
            "  ", " ")

        return result

    def remove_unwanted_tokens_text_col(tltuple):
        text, lang = tltuple
        if type(text) != str:
            return text
        prem_tokens = [tok for tok in text.replace("?", " ").split() if not (is_token_css(tok) or is_token_latex(tok))]
        text = " ".join(prem_tokens[:min(len(prem_tokens), 200)])
        # first reduce websites and files info, and then remove unwanted tokens
        good_tokens = [tok for tok in transform_websites_and_files(text) if not
        (is_token_code(tok, lang) or is_token_source(tok) or is_token_version(tok) or is_token_bad_token(tok, lang))]
        if len(good_tokens) == 0:
            return np.nan
        result = " ".join(good_tokens).replace("( file resource: )", "").replace("( web resource: )", "").replace("_",
                                                                                                                  " ").replace(
            "  ", " ")

        lang_res = lang_detect_model.classify(result)
        detect_lang = lang_res[0]
        prob = lang_res[1]
        if (((detect_lang == lang) and prob > 0.75) or ((detect_lang == "en") and prob > 0.75)):
            return result

        return np.nan

    # transform long links and files names into a good form
    def remove_unwanted_tokens_bulk(text_column, lang_column, remover_func):
        new_col = np.empty(shape=(len(text_column)), dtype="object")
        for k in range(len(text_column)):
            if k % 20000 == 0:
                print(k, "completed out of ", len(text_column))
            text = text_column.iloc[k]
            lang = lang_column.iloc[k]
            new_col[k] = remover_func((text, lang))
        return pd.Series(data=new_col, index=text_column.index)

    def merge_into_description(dttuple):
        description, text = dttuple
        if type(description) != str:
            return text
        if type(text) != str:
            return description
        desctokens = description.split()
        texttokens = text.split()
        if len(desctokens) < 10:
            return " ".join(desctokens + texttokens[:min(len(texttokens), 40)])
        return description

    def merge_contents_description_bulk(description_column, text_column):
        new_col = np.empty(shape=(len(text_column)), dtype="object")
        for k in range(len(text_column)):
            if k % 20000 == 0:
                print(k, "completed out of ", len(text_column))
            description = description_column.iloc[k]
            text = text_column.iloc[k]
            new_col[k] = merge_into_description((description, text))
        return pd.Series(data=new_col, index=text_column.index)

    def is_bad_description(x):
        if type(x) != str:
            return True
        return len(x.split()) < 10

    def preprocess_before_translation():
        data_contents.loc[(data_contents["description"] == data_contents["title"]) & (
            ~data_contents["text"].isnull()), "description"] = np.nan

        print("Removing usual unwanted tokens....")
        data_topics["description"] = remove_unwanted_tokens_bulk(data_topics["description"], data_topics["language"],
                                                                 remove_unwanted_tokens)
        data_topics["title"] = remove_unwanted_tokens_bulk(data_topics["title"], data_topics["language"],
                                                           remove_unwanted_tokens)
        data_contents["description"] = remove_unwanted_tokens_bulk(data_contents["description"],
                                                                   data_contents["language"], remove_unwanted_tokens)
        data_contents["title"] = remove_unwanted_tokens_bulk(data_contents["title"], data_contents["language"],
                                                             remove_unwanted_tokens)

        print("Removing text unwanted tokens....")
        bad_locs = data_contents["description"].apply(is_bad_description)
        new_text_col = remove_unwanted_tokens_bulk(data_contents.loc[bad_locs, "text"],
                                                   data_contents.loc[bad_locs, "language"],
                                                   remove_unwanted_tokens_text_col)

        print("Merging contents into description....")
        data_contents.loc[bad_locs, "description"] = merge_contents_description_bulk(
            data_contents.loc[bad_locs, "description"], new_text_col)

        # impute description with title
        impute_rows = data_contents["description"].isnull() & (~data_contents["title"].isnull())
        data_contents.loc[impute_rows, "description"] = data_contents.loc[impute_rows, "title"]
        impute_rows = data_topics["description"].isnull() & (~data_topics["title"].isnull())
        data_topics.loc[impute_rows, "description"] = data_topics.loc[impute_rows, "title"]
        print("All completed!")

    preprocess_before_translation()

    if not os.path.isdir("generated_data/"):
        os.mkdir("generated_data/")

    data_topics.to_csv("generated_data/new_topics.csv")
    data_contents.to_csv("generated_data/new_contents.csv")

def translate_subprocess():
    import autocorrect
    import os
    import sys
    import translate_helpers
    import pandas as pd
    import numpy as np
    import gc
    import time

    data_topics = pd.read_csv("generated_data/new_topics.csv", index_col=0)
    data_contents = pd.read_csv("generated_data/new_contents.csv", index_col=0)

    def translate_languages(batch_size=1000):
        # -------------- detect actual languages --------------
        data_topics.loc[data_topics["language"] == "swa", "language"] = "sw"
        data_contents.loc[data_contents["language"] == "swa", "language"] = "sw"

        ctime = time.time()
        topics_lang = translate_helpers.obtain_language_info(data_topics)
        contents_lang = translate_helpers.obtain_language_info(data_contents)
        ctime = time.time() - ctime
        print("Obtaining language info: ", ctime)
        """display((topics_lang["description_lang"] != data_topics["language"]).sum())
        display((topics_lang["title_lang"] != data_topics["language"]).sum())
        display((contents_lang["description_lang"] != data_contents["language"]).sum())
        display((contents_lang["title_lang"] != data_contents["language"]).sum()) """

        topics_lang["description_idx"] = np.arange(len(topics_lang))
        topics_lang["title_idx"] = np.arange(len(topics_lang), 2 * len(topics_lang))
        contents_lang["description_idx"] = np.arange(2 * len(topics_lang),
                                                     len(contents_lang) + 2 * len(topics_lang))
        contents_lang["title_idx"] = np.arange(len(contents_lang) + 2 * len(topics_lang),
                                               2 * len(contents_lang) + 2 * len(topics_lang))

        topics_lang.loc[data_topics["description"].isnull(), "description_lang"] = "none"
        topics_lang.loc[data_topics["title"].isnull(), "title_lang"] = "none"
        contents_lang.loc[data_contents["description"].isnull(), "description_lang"] = "none"
        contents_lang.loc[data_contents["title"].isnull(), "title_lang"] = "none"

        mtext = np.empty(shape=(2 * len(topics_lang) + 2 * len(contents_lang)), dtype="object")
        mlang = np.empty(shape=(2 * len(topics_lang) + 2 * len(contents_lang)), dtype="object")

        mlang[:len(topics_lang)] = topics_lang["description_lang"]
        mtext[:len(topics_lang)] = data_topics["description"]

        mlang[len(topics_lang): (2 * len(topics_lang))] = topics_lang["title_lang"]
        mtext[len(topics_lang): (2 * len(topics_lang))] = data_topics["title"]

        mlang[(2 * len(topics_lang)):(len(contents_lang) + 2 * len(topics_lang))] = contents_lang[
            "description_lang"]
        mtext[(2 * len(topics_lang)):(len(contents_lang) + 2 * len(topics_lang))] = data_contents["description"]

        mlang[(len(contents_lang) + 2 * len(topics_lang)):(2 * len(contents_lang) + 2 * len(topics_lang))] = \
        contents_lang["title_lang"]
        mtext[(len(contents_lang) + 2 * len(topics_lang)):(2 * len(contents_lang) + 2 * len(topics_lang))] = \
        data_contents["title"]

        total_text = pd.DataFrame(index=np.arange(2 * len(contents_lang) + 2 * len(topics_lang)),
                                  data={"lang": mlang, "text": mtext})

        total_text.to_csv("generated_data/total_text_before_translate.csv")

        # -------------- do translation here --------------

        languages = total_text.lang.value_counts().index

        langs_to_translate = list(pd.Index(translate_helpers.list_available_langs()).intersection(languages))
        langs_to_translate.sort()
        translated_results = np.empty(shape=len(total_text), dtype="object")
        translated_results[:] = np.nan

        ctime = time.time()
        for language in langs_to_translate:
            translate_info = translate_helpers.load_model(language)
            lang_idx = total_text.loc[total_text["lang"] == language].index
            length = len(lang_idx)

            batch_size = 1024
            tlow = 0
            continuous_success = 0
            while tlow < length:
                thigh = min(tlow + batch_size, length)
                try:
                    text_idx = lang_idx[np.arange(tlow, thigh)]
                    sentences = list(total_text.loc[text_idx, "text"])
                    translated_sentences = translate_helpers.translate_sentences(sentences, translate_info)
                    translated_results[np.array(text_idx, dtype=np.int64)] = translated_sentences

                    tlow = thigh
                    continuous_success += 1
                    if continuous_success == 10:
                        continuous_success = 0
                        batch_size = max(int(batch_size * 1.7), batch_size + 1)
                except RuntimeError as err:
                    if not "CUDA failed with error out of memory" in str(err):
                        raise err
                    batch_size = max(int(batch_size / 1.3), 1)
                    continuous_success = 0
                    gc.collect()

            del translate_info
            print("Translated ", language)
        total_text["text_translate"] = translated_results
        total_text.loc[total_text["lang"] == "en", "text_translate"] = total_text.loc[
            total_text["lang"] == "en", "text"]
        ctime = time.time() - ctime
        print("Translation time used: ", ctime)
        # total_text.to_csv("total_text.csv")

        word_corrector = autocorrect.Speller(fast=True)
        ctime = time.time()
        total_text["text_translate"] = total_text["text_translate"].map(
            lambda text: word_corrector(text) if type(text) == str else text)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #    corrected = executor.map(lambda text: word_corrector(text) if type(text)==str else text, total_text["text_translate"].tolist())
        ctime = time.time() - ctime
        print("Word correction time used: ", ctime)
        # total_text["text_translate"] = pd.Series(data = corrected, index = total_text.index)
        # total_text.to_csv("total_text_corrected.csv")

        topics_lang["description"] = np.array(total_text.loc[np.arange(len(topics_lang)), "text_translate"])
        topics_lang["title"] = np.array(
            total_text.loc[np.arange(len(topics_lang), 2 * len(topics_lang)), "text_translate"])
        contents_lang["description"] = np.array(total_text.loc[np.arange(2 * len(topics_lang),
                                                                         len(contents_lang) + 2 * len(
                                                                             topics_lang)), "text_translate"])
        contents_lang["title"] = np.array(total_text.loc[np.arange(len(contents_lang) + 2 * len(topics_lang),
                                                                   2 * len(contents_lang) + 2 * len(
                                                                       topics_lang)), "text_translate"])

        del total_text

        return topics_lang, contents_lang

    ctime = time.time()
    topics_translated, contents_translated = translate_languages(8000)
    ctime = time.time() - ctime
    print("Total translation time used:", ctime)

    data_topicsn = data_topics.join(topics_translated[["description", "title"]], rsuffix="_translate")
    data_contentsn = data_contents.join(contents_translated[["description", "title"]], rsuffix="_translate")

    os.remove("generated_data/new_topics.csv")
    os.remove("generated_data/new_contents.csv")

    data_topicsn.to_csv("generated_data/new_topics.csv")
    data_contentsn.to_csv("generated_data/new_contents.csv")