import ctranslate2
import ctranslate2.converters
import os
import zipfile
import shutil

# unzip the files here
files = os.listdir("data/opus_translation_models")

langs = []

for lang_file in files:
    if lang_file.endswith(".zip"):
        lang = lang_file[:2]
        extract_dir = "data/opus_translation_models/" + lang + "-en"
        if not os.path.isdir(extract_dir):
            os.mkdir(extract_dir)
            with zipfile.ZipFile("data/opus_translation_models/" + lang_file, "r") as zip_file:
                zip_file.extractall(extract_dir)
        langs.append(lang)

print(langs)

# convert the files here
print("Converting files....")
for lang in langs:
    source_opus_model_dir = "data/opus_translation_models/" + lang + "-en"
    target_dir = "data/ct2_models_from_opus/" + lang + "-en"
    if not os.path.isdir(target_dir):
        print("Converting " + lang + "...")
        converter = ctranslate2.converters.OpusMTConverter(model_dir = source_opus_model_dir)
        converter.convert(target_dir)
        print("Conversion of lang "+ lang + " success!")

# copy the sentencepiece sources to directory
print("Copying files:")
for lang in langs:
    source_opus_model_spm = "data/opus_translation_models/" + lang + "-en/source.spm"
    target_file = "data/ct2_models_from_opus/" + lang + "-en/source.spm"
    if not os.path.isfile(source_opus_model_spm):
        source_opus_model_spm = "data/opus_translation_models/" + lang + "-en/source.bpe"
        target_file = "data/ct2_models_from_opus/" + lang + "-en/source.bpe"
    if not os.path.isfile(target_file):
        shutil.copyfile(source_opus_model_spm, target_file)
        print("Copying file " + lang + " success!")