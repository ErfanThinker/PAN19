import codecs
import glob
import os
from collections import defaultdict

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


def read_files(path, label, gt_dict=None):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path + os.sep + label + os.sep + '*.txt')
    files.sort()
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        file_name = v.rsplit(os.sep, maxsplit=1)[1]
        label = label if gt_dict is None else gt_dict[file_name]
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
                                                   path_to_jar=stanford_res_path + "stanford-postagger.jar", java_options='-mx3g')
        results = tagger.tag(word_tokenize(texts, language=lang_map[lang]))
        # del tagger
        # del stanford_tagger
        if lang == 'en':  # convert eng tags to universal tags
            results = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in results]

    return results


def clean_folder(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path):
            #     shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def represent_text(text, n):
    # Extracts all character 'n'-grams from  a 'text'
    if n > 0:
        tokens = [text[i:i + n] for i in range(len(text) - n + 1)]
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency


def extract_n_grams(texts, n, ft):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences = defaultdict(int)
    for (text, label) in texts:
        text_occurrences = represent_text(text, n)
        for ngram in text_occurrences:
            occurrences[ngram] += text_occurrences[ngram]
            # if ngram in occurrences:
            #     occurrences[ngram]+=text_occurrences[ngram]
            # else:
            #     occurrences[ngram]=text_occurrences[ngram]
    vocabulary = []
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)
    return vocabulary


def shuffle_docs(texts, labels):
    new_texts = []
    new_labels = []
    for ind, text in enumerate(texts):
        if ind != len(texts) - 1:
            tokenized_text = text.split(' ')
            first_half = tokenized_text[:int(len(tokenized_text) / 2)]
            second_half = tokenized_text[int(len(tokenized_text) / 2):]
            remaining_text_from_this_writer = [text for index, text in enumerate(texts)
                                               if index > ind and labels[index] == labels[ind]]
            for other_text in remaining_text_from_this_writer:
                tokenized_other_text = other_text.split(' ')
                first_half_of_other_text = tokenized_other_text[:int(len(tokenized_other_text) / 2)]
                second_half_of_other_text = tokenized_other_text[int(len(tokenized_other_text) / 2):]
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(first_half) + ' ' + ' '.join(second_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(first_half_of_other_text))
                new_texts.append(' '.join(second_half) + ' ' + ' '.join(second_half_of_other_text))
                new_labels.extend([labels[ind]] * 4)

    new_texts.extend(texts)
    new_labels.extend(labels)
    return new_texts, new_labels
