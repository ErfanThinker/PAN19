import argparse
import codecs
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict

from nltk import map_tag
from nltk import word_tokenize

from geneticApproach import ABALGA as geneticSolution
from geneticApproach import MyDataParser as dp

stanford_lang_models = {"en": "english-bidirectional-distsim.tagger",
                        "sp": "spanish-ud.tagger",
                        "fr": "french-ud.tagger",
                        "it": "italian-fast.ud.tagger"
                        }
lang_map = {'en': 'english',
            'sp': 'spanish',
            'fr': 'french',
            'it': 'italian'}
stanford_res_path = ".." + os.sep + "models" + os.sep


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


def process_doc(train_texts, idx, lang, lock, word_count, train_word_set):
    words_plus_pos = extract_words_plus_pos_tags(train_texts[idx], lang)
    lock.acquire()
    train_texts[idx] = words_plus_pos
    for tp in words_plus_pos:
        word_count[tp] += 1
    train_word_set.update(words_plus_pos)
    lock.release()


def generate_scores(problem_dir, lang):
    infoproblem = problem_dir + os.sep + 'problem-info.json'
    candidates = []
    pool = mp.Pool(mp.cpu_count() - 2)
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    train_docs = []
    for candidate in candidates:
        train_docs.extend(pool.apply_async(read_files, (problem_dir, candidate)))
    pool.close()
    pool.join()
    train_texts = [text for i, (text, label) in enumerate(train_docs)]
    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    train_word_set = set()
    word_count = defaultdict(int)
    lock = mp.Lock()
    for idx, train_text in enumerate(train_texts):
        pool.apply_async(process_doc, (train_texts, idx, lang, lock, word_count, train_word_set))
    pool.close()
    pool.join()

    train_word_set = [tp for tp in train_word_set if word_count[tp] > 1]
    maxAge = 50
    poolSize = 500

    print("GeneSetLength:", str(len(train_word_set)))
    print("Running the algorithm...")
    for trainWordScores, percentage, totalSeconds in geneticSolution.justDoIt(train_word_set, train_texts, train_labels,
                                                                              maxAge=maxAge, poolSize=poolSize):
        print("\n\rpercentage:", str(percentage), 'time:', str(totalSeconds))
        dp.writeWordScores(
            'Word_Scores_' + problem_dir.rsplit(os.path)[1] + '_poolSize_' + str(poolSize) + '_maxAge_' + str(maxAge) +
            '__percentage__' + str(percentage) + '_totalSeconds_' + str(totalSeconds.total_seconds())
            , trainWordScores)


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PAN-19 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    args = parser.parse_args()
    if not args.i:
        args.i = '..' + os.sep + 'pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23' + os.sep + 'problem00002'
    lang = 'en'
    generate_scores(args.i, lang)


if __name__ == '__main__':
    main()
