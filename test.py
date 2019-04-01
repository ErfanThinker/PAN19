# coding=utf-8
import logging
import pprint

import nltk.tag.stanford as stanford_tagger
from nltk import *

# from treetagger import TreeTagger


# download()

lang_models = {"en": "english-bidirectional-distsim.tagger",
               "sp": "spanish-ud.tagger",
               "fr": "french-ud.tagger"}
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pp = pprint.PrettyPrinter(indent=4)
samples = ["Human machine interface for lab abc computer applications",
           "A survey of user opinion of computer system response time",
           "The EPS user interface management system",
           "System and human system engineering testing of EPS",
           "Relation of user perceived response time to error measurement",
           "The generation of random binary unordered trees",
           "The intersection graph of paths in trees",
           "Graph minors IV Widths of trees and well quasi ordering",
           "Graph minors A survey",
           "De... Derek, no te vayas - llamó en un susurro, la voz temblándole tanto como las manos y un peso en el estómago que amenazaba con hundirlo mientras los ojos se le llenaban de lágrimas - Der, por fav...  empezó de nuevo, porque no era capaz de girarse, no era capaz de girarse para acercarse él y ver como su novio seguía andando dándole la espalda, o como directamente cambiaba de trayectoria y se iba hacia la puerta, que era lo que se él se merecía, pero no acabó de hablar siquiera, porque los brazos del moreno envolviéndolo cortaron sus palabras que acabaron en suspiro.",
           "cómo la enfrentó, le hizo ver qué era lo que realmente tenía que hacer. El capitán que nadie veía como un jefe pero todos lo sentían como un líder, al final. Ella también lo terminó por ver así. Esa decisión, esa mirada (Vivi se sentía atravesada por ella, igual que si él hubiera cogido sus sentimientos y los hubiera leído como si de un libro se tratara) cuando sabía que había que hacer algo. La confianza infinita en sí mismo y en sus nakama."

           ]
tagger = stanford_tagger.StanfordPOSTagger("." + os.sep + "models" + os.sep + "spanish-ud.tagger", path_to_jar="." + os.sep + "models" + os.sep + "stanford-postagger.jar")

pp.pprint(tagger.tag(word_tokenize(samples[-1], language="spanish")))
