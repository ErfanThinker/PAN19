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

def read_files(path, label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path + os.sep + label + os.sep + '*.txt')
    files.sort()
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
