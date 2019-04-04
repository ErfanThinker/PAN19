from __future__ import print_function

import argparse
import codecs
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict

from nltk import map_tag
from nltk import word_tokenize

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


def tuple_list_2_dict(da_list):
    out = defaultdict(list)
    for val in da_list:
        out[val[1]].append(val[0])
    return out


def create_author_2_word_pos_dict(authors, texts):
    out = defaultdict(list)
    for idx, aut in enumerate(authors):
        out[aut].extend(texts[idx])
    return out


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


def generate_scores(problem_dir, lang):
    infoproblem = problem_dir + os.sep + 'problem-info.json'
    candidates = []
    pool = mp.Pool(int(mp.cpu_count() / 2) - 1)
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    train_docs = []
    for candidate in candidates:
        train_docs.extend(read_files(problem_dir, candidate))
    train_texts = [(text, lang, i) for i, (text, label) in enumerate(train_docs)]
    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    train_word_set = set()
    print("doc count to process: ", str(len(train_texts)))

    train_texts = pool.map(process_doc, train_texts)
    print('process_doc, done!')
    for train_text in train_texts:
        train_word_set.update(train_text)

    # print(train_texts)
    print(str(len(train_word_set)))
    # This will give something like this: {label_0:[text_0, text_1,...], label_1: [...] , ... }
    author_dict = create_author_2_word_pos_dict(train_labels, train_texts)
    # ppr = pp.PrettyPrinter(indent=4)
    # ppr.pprint(author_dict)
    print(author_dict)


def main():
    parser = argparse.ArgumentParser(description='PAN-19 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    args = parser.parse_args()
    if not args.i:
        args.i = '.' + os.sep + 'pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23' + os.sep + 'problem00002'
    lang = 'en'
    generate_scores(args.i, lang)


if __name__ == '__main__':
    main()
