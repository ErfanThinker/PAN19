import codecs
import glob
import os

from nltk import word_tokenize, map_tag

stanford_lang_models = {"en": "english-bidirectional-distsim.tagger",
                        "sp": "spanish-ud.tagger",
                        "fr": "french-ud.tagger",
                        "it": "italian-fast.ud.tagger"
                        }
lang_map = {'en': 'english',
            'sp': 'spanish',
            'fr': 'french',
            'it': 'italian'}

stanford_res_path = "." + os.sep + "models" + os.sep


def read_files(path, label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path + os.sep + label + os.sep + '*.txt')
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        f.close()
    return texts


def process_doc(train_text):  # [(text, lang, i) , ... ]
    print("Processing doc #", str(train_text[2] + 1))
    return extract_words_plus_pos_tags(train_text[0], train_text[1])


def extract_words_plus_pos_tags(texts, lang):
    results = []
    if lang in stanford_lang_models:
        import nltk.tag.stanford as stanford_tagger
        tagger = stanford_tagger.StanfordPOSTagger(stanford_res_path + stanford_lang_models[lang],
                                                   path_to_jar=stanford_res_path + "stanford-postagger.jar")
        results = tagger.tag(word_tokenize(texts, language=lang_map[lang]))
        if lang == 'en':  # convert eng tags to universal tags
            results = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in results]

    return results
