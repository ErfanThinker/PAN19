# coding=utf-8
import codecs
import logging
import pprint as pp

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


def vectorize_file(path, lang):
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
        return results
    # else: #to process it text
    #     import treetaggerwrapper
    #     tagger = treetaggerwrapper.TreeTagger(TAGLANG=lang)
    #     tags = treetaggerwrapper.make_tags(tagger.tag_text(texts), exclude_nottags=True)
    #     for element in tags:
    #         if isinstance(element, treetaggerwrapper.Tag) :
    #             results.append((element.word, element.pos))
    # results = [(word, map_tag('es-treetagger', 'universal', tag)) for word, tag in results ]
    return results


test_result = vectorize_file('pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23' + os.sep +
                             'problem00012' + os.sep + 'candidate00002' + os.sep + 'known00001.txt',
                             'it')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
ppr = pp.PrettyPrinter(indent=4)
ppr.pprint(test_result)
# pp.pprint(mapping._UNIVERSAL_TAGS)
