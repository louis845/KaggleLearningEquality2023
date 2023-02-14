# Python packages used
## Python version 3.10
## Core packages:
* numpy 1.21.6
* pandas 1.3.5
* scipy 1.7.3
* matplotlib 3.5.2
* scikit-learn 1.0.2
* seaborn 0.11.2
* cupy-cuda11x 11.5.0 (only for data_bert_gpu.py)
## Packages for natural language processing:
These packages are used for translation (to English), and pretrained transformer models for NLP tasks.
* Previous used packages (used in model_vectorizer_simple):
  * argostranslate 1.7.5 (don't need to install this locally since this would be used in Kaggle notebook)
  This package is used for lemmatizing (grouping) English words.
    * All languages -> en downloaded
  * nltk
    * WordNet
    * averaged_perceptron_tagger
* Currently used packages (BERT):
  * BERT transformer model https://www.tensorflow.org/text/tutorials/classify_text_with_bert
  * Opus-MT pretrained models https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/models (Creative Commons Attribution 4.0)
  * ctranslate2 2.24.0 https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE (MIT License)
  * sentencepiece 0.1.96 https://github.com/google/sentencepiece (Apache License 2.0)
  * langdetect 1.0.9 https://github.com/Mimino666/langdetect (Apache License 2.0)