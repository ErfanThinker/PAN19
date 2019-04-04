# coding=utf-8
import codecs
import logging
import pprint as pp

import numpy as np
from nltk import *

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


def extract_pos_tags(path, lang):
    f = codecs.open(path, 'r', encoding='utf-8')
    texts = f.read()
    f.close()
    results = []
    if lang in stanford_lang_models:
        import nltk.tag.stanford as stanford_tagger
        tagger = stanford_tagger.StanfordPOSTagger(stanford_res_path + stanford_lang_models[lang],
                                                   path_to_jar=stanford_res_path + "stanford-postagger.jar")
        # sentences = sent_tokenize(texts, language=lang_map[lang])
        # for sentence in sentences:
        #     results.extend(tagger.tag(word_tokenize(sentence, language=lang_map[lang])))
        results = tagger.tag(word_tokenize(texts, language=lang_map[lang]))
        if lang == 'en':  # convert eng tags to universal tags
            results = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in results]
    # else: #to process it text
    #     import treetaggerwrapper
    #     tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)
    #     tags = treetaggerwrapper.make_tags(tagger.tag_text(texts), exclude_nottags=True)
    #     for element in tags:
    #         if isinstance(element, treetaggerwrapper.Tag) :
    #             results.append((element.word, element.pos))
    # results = [(word, map_tag('es-treetagger', 'universal', tag)) for word, tag in results ]
    return np.asanyarray(results)


# This class creates a word -> index mapping (e.g,. "dad/pos" -> 5) and vice-versa
# (e.g., 5 -> "dad/pos") for each language,
class LanguageIndex():
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for v, pos in self.vocab_list:
            self.vocab.update(v + '/' + pos)

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


test_result = extract_pos_tags('pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23' + os.sep +
                             'problem00012' + os.sep + 'candidate00002' + os.sep + 'known00001.txt',
                             'it')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
ppr = pp.PrettyPrinter(indent=4)
ppr.pprint(test_result)
print(test_result.shape)
# pp.pprint(mapping._UNIVERSAL_TAGS)
