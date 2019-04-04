import copy
import datetime
import multiprocessing as mp
import random
import sys

import numpy as np
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from geneticApproach import Genetics as gn


def calc_text_score(genes, text, features, idx):
    localSum = 0
    for ind, word in enumerate(text):
        localSum += genes[word]
    features[idx] = localSum


def get_fitness(genes, texts, labels):
    features = np.asarray([0 for _ in labels])
    acc = 0
    pool = mp.Pool(mp.cpu_count() - 2)
    for idx, text in enumerate(texts):
        pool.apply_async(calc_text_score, (genes, text, features, idx))
    # calculate accuracy
    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_train_data = max_abs_scaler.fit_transform(features)
    skf = StratifiedKFold(5, True, 2019)
    missfits = []
    for train, test in skf.split(features, labels):
        clf = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1)))
        train_labels = [l for idx, l in enumerate(labels) if idx in train]
        test_labels = [l for idx, l in enumerate(labels) if idx in test]
        clf.fit(scaled_train_data[train], train_labels)
        predictions = clf.predict(scaled_train_data[test])
        proba = clf.predict_proba(scaled_train_data[test])
        # Reject option (used in open-set cases)
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < 0.1:
                predictions[i] = u'<UNK>'
        missfits.extend([idx for idx, v in enumerate(labels) if v != labels[scaled_train_data in train][idx]])
    return acc, missfits


def mutate(genes, sortedGeneset, fitness, trainCommentsForAlga):
    allOfThem = []
    for sentInd in [index for index, value in enumerate(fitness.LabelFitness) if not value == 0]:
        allOfThem.extend([term[1] for term in trainCommentsForAlga[sentInd][5:]])

    # startTime = datetime.datetime.now()
    candidateWordsSet = set(allOfThem)
    candidateGenes = {k: v for k, v in genes.items() if k in candidateWordsSet}
    # print('candidateWordsSet', str(datetime.datetime.now() - startTime))
    # startTime = datetime.datetime.now()
    # candidateWordCountHolder = {val: 0 for val in candidateWordsSet}
    # for word in candidateWordsSet:
    #     candidateWordCountHolder[word] = allOfThem.count(word)

    # betterWords = sorted(candidateWordCountHolder, key=lambda x : candidateWordCountHolder[x], reverse=True)
    # print('betterWords', str(datetime.datetime.now() - startTime))

    numOfChanges = 2 if random.randint(0, 4) == 0 and len(candidateGenes.keys()) > 1 else 1
    randomResult = random.randint(0, 4)
    if randomResult == 0:  # 10% chance
        indexes = random.sample(genes.keys(), random.randint(1, len(genes.keys()))) if random.randint(0, 2) == 0 \
            else random.sample(genes.keys(), numOfChanges)
    # elif len(betterWords) >=  1 and 1 <= randomResult <= 3:   #30% chance
    #     indexes =  betterWords[0:numOfChanges] if len(betterWords) >= numOfChanges else [betterWords[0]]
    else:  # 60% chance
        indexes = random.sample(candidateGenes.keys(), numOfChanges)

    while len(indexes) > 0:
        index = indexes.pop()
        toReplace = sortedGeneset.getSample(count=2)
        if genes[index] < 0:
            toReplace[0] = 0 - toReplace[0]
            toReplace[1] = 0 - toReplace[1]
        genes[index] = toReplace[1] if genes[index] == toReplace[0] else toReplace[0]
        genes[index] = 0 - genes[index] if random.randint(0, 3) == 0 else genes[index]


def crossover(parentGenes, donorGenes, initialFitness, donorFitness, fnGetFitness):
    identical = True
    candidateGeneIndices = []
    for key in parentGenes.keys():
        if parentGenes[key] != donorGenes[key]:
            identical = False
            candidateGeneIndices.append(key)
    if identical:
        return None

    childGenes = copy.deepcopy(parentGenes)

    # if initialFitness < donorFitness:
    #     randomSelection = random.sample(candidateGeneIndices,
    #                                     random.randint(1, min(10, len(candidateGeneIndices) - 1))
    #                                     if len(candidateGeneIndices) > 1 else 1)
    #     for key in randomSelection:
    #         childGenes[key] = donorGenes[key]
    #     return childGenes

    maxAttempts = random.randint(1, 10) if len(candidateGeneIndices) > 1 else 1
    for attempt in range(maxAttempts):
        hasImprovement = False
        for key in random.sample(candidateGeneIndices,
                                 random.randint(1, max(2, round(len(candidateGeneIndices) / 100)))
                                 if len(candidateGeneIndices) > 1 else 1):
            pastVal = childGenes[key]
            childGenes[key] = donorGenes[key]
            if initialFitness > fnGetFitness(childGenes):
                childGenes[key] = pastVal
            else:
                hasImprovement = True

        if hasImprovement:
            return childGenes
        else:
            continue
    return childGenes


# def crossover(parentGenes, donorGenes, initialFitness, donorFitness):
#     identical = True
#     bestGeneChoice = copy.deepcopy(parentGenes)
#     for key in parentGenes.keys():
#         if parentGenes[key] != donorGenes[key]:
#             useDonorGene = True if random.uniform(0, initialFitness + donorFitness) <=  float(donorFitness) else False
#             useDonorGene = not useDonorGene if initialFitness < donorFitness else useDonorGene
#             bestGeneChoice[key] = donorGenes[key] if useDonorGene else bestGeneChoice[key]
#             identical = False
#
#     return bestGeneChoice if not identical else None

def create(wordSet, geneSet):
    return {word: geneSet.getSample(pastValue=None)[0] for word in wordSet}


def display(candidate, startTime, poolSize, maxAge):
    timeDiff = str(datetime.datetime.now() - startTime)
    fitness = candidate.Fitness
    sys.stdout.write("\r(%d,%d) Fitness(Strategy: %s) %s \t%s" % (
        poolSize, maxAge,
        candidate.Strategy.name,
        fitness,
        timeDiff
    ))
    sys.stdout.flush()


class Fitness:
    def __init__(self, acc, missfits):
        self.acc = acc
        self.missfits = missfits

    def __gt__(self, other):
        return self.acc > other.acc

    def __lt__(self, other):
        return self.acc < other.acc

    def __eq__(self, other):
        # return abs(self.Difference) == abs(other.Difference)
        return self.acc == other.acc

    def __str__(self):
        return "Acc: {:0.2f}, miss-fits: %d".format(float(self.acc), self.missfits)


class GeneSet:
    def __init__(self):
        pass

    def getSample(self, pastValue=None, count=1):
        samples = []
        if pastValue is None:
            for _ in range(count):
                while True:
                    rand = random.randint(0, 10)
                    samples.append(rand)
        else:
            rand = pastValue
            while rand == pastValue:
                rand = random.randint(0, 10)
            samples.append(rand)
        return samples


def justDoIt(trainWordSet, texts, labels, maxAge=50, poolSize=20
             ):
    random.seed(2019)
    geneSet = GeneSet()  # [0, 10]

    startTime = datetime.datetime.now()

    def fnDisplay(candidate):
        display(candidate, startTime, poolSize, maxAge)

    def fnGetFitness(genes):
        return get_fitness(genes, texts, labels)

    def fnMutate(genes, fitness):
        mutate(genes, geneSet, fitness, trainWordSet)

    def fnCreate():
        return create(trainWordSet, geneSet)

    def fnCrossover(parentGenes, donorGenes, parentFitness, donorFitness):
        return crossover(parentGenes, donorGenes, parentFitness, donorFitness, fnGetFitness)

    optimalFitness = Fitness(100)

    for best, percentage in gn.get_best(fnGetFitness, 0, optimalFitness,
                                        geneSet, fnDisplay, custom_mutate=fnMutate, custom_create=fnCreate,
                                        maxAge=maxAge,
                                        poolSize=poolSize, crossover=fnCrossover):
        yield best.Genes, percentage, datetime.datetime.now() - startTime
