import random
import copy
import nltk
import sys
from nltk import BigramCollocationFinder
from nltk import BigramAssocMeasures
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn

import MyDataParser

valuedPoses = ["JJ", "JJR", "JJS", "NN", "NNS", "VB", "VBD", "VBG", "VBP", "VBN", "VBZ", "RB", "RBR", "RBS", "CD"]
negations = ['not', 'n\'t', 'no', 'barely', 'hardly', 'rarely', 'never']


def get_window(tokenizedSentenceUnigram, aspect, windowSize):
    # tokenizedLemmatizedSentenceWithDetails -> [ [index, word, pos] , .... ]
    startIndexOfAspect = -1
    stopIndexofAspect = -1
    if nltk.word_tokenize(aspect[0])[0] in tokenizedSentenceUnigram:
        startIndexOfAspect = tokenizedSentenceUnigram.index(nltk.word_tokenize(aspect[0])[0])
    if not startIndexOfAspect == -1 and nltk.word_tokenize(aspect[0])[-1] in tokenizedSentenceUnigram[
                                                                             startIndexOfAspect + 1:]:
        stopIndexofAspect = startIndexOfAspect + tokenizedSentenceUnigram[startIndexOfAspect:].index(
            nltk.word_tokenize(aspect[0])[-1])
    else:
        startOfAspectTerm = nltk.word_tokenize(aspect[0])[0]
        stopOfAspectTerm = nltk.word_tokenize(aspect[0])[-1]
        for word in tokenizedSentenceUnigram:
            if startOfAspectTerm in word:
                startIndexOfAspect = tokenizedSentenceUnigram.index(word)
            if not startIndexOfAspect == -1 and stopOfAspectTerm in word:
                stopIndexofAspect = tokenizedSentenceUnigram.index(word)
                aspectInSentence = word

    if not stopIndexofAspect == -1:
        window = tokenizedSentenceUnigram[
                 max(0, startIndexOfAspect - windowSize): min((len(tokenizedSentenceUnigram) - 1),
                                                              stopIndexofAspect + windowSize) + 1]
    else:
        window = tokenizedSentenceUnigram[
                 max(0, startIndexOfAspect - windowSize): min((len(tokenizedSentenceUnigram) - 1),
                                                              startIndexOfAspect + windowSize) + 1]

    # remove aspect term from window
    for word in nltk.word_tokenize(aspect[0]):
        if (word in window):
            window.remove(word)
        else:
            for ind, w in enumerate(window):
                if word in w:
                    window.remove(w)
                    break

    return window


def findClosestAspect(tokenizedSentence, wordStruct, aspects):
    indices = [t[0] for t in tokenizedSentence]
    terms = [t[1] for t in tokenizedSentence]
    wordIndex = wordStruct[0]
    closestAspect = None
    intendedLabel = 1
    closestDistance = 10000
    aspectsRevised = list(aspects.values()) if type(aspects) is dict else aspects
    for aspect in aspectsRevised:
        if type(aspects) is list or (type(aspects) is dict and not aspect[2]):  # aspect is explicit
            startIndexOfAspect = -1
            stopIndexofAspect = -1
            tokenizedAspect = nltk.word_tokenize(aspect[0])
            if len(tokenizedAspect) == 1:
                if (tokenizedAspect[0] in terms):
                    stopIndexofAspect = startIndexOfAspect = indices[terms.index(tokenizedAspect[0])]
            else:
                startOfAspectTerm = tokenizedAspect[0]
                indexDiff = len(tokenizedAspect) - 1
                stopOfAspectTerm = tokenizedAspect[-1]
                for ind, word in enumerate(terms):
                    try:
                        check = stopOfAspectTerm == terms[ind + indexDiff]
                        if startIndexOfAspect == -1 and startOfAspectTerm == word and stopOfAspectTerm == terms[
                            ind + indexDiff]:
                            startIndexOfAspect = indices[ind]
                            stopIndexofAspect = indices[ind + indexDiff]
                            break
                    except IndexError:
                        print('alaki')
                        print(tokenizedSentence)
                        print(wordStruct)
                        print(aspects)
                        print('-------------------')
                        break

            dist = abs(wordIndex - startIndexOfAspect) if wordIndex <= startIndexOfAspect else abs(
                wordIndex - stopIndexofAspect)
            if dist < closestDistance:
                closestDistance = dist
                closestAspect = aspect
    if closestAspect is None:
        closestAspect = list(filter(lambda x: x[2] is True, list(aspects.values())))[0]
    intendedLabel = closestAspect[1]
    if (str(wordStruct[2]).startswith('J') or str(wordStruct[2]).startswith('V')) and \
            tokenizedSentence[indices.index(wordIndex) - 1][1] in negations:
        intendedLabel = -intendedLabel
    return closestAspect, intendedLabel


def getWindowBeta(tokenizedSentence, aspect, aspects, windowSize):
    indices = [t[0] for t in tokenizedSentence]
    terms = [t[1] for t in tokenizedSentence]
    isImplicit = False
    # try:
    if not aspect[2]:
        startIndexOfAspect = -1
        stopIndexofAspect = -1
        tokenizedAspect = nltk.word_tokenize(aspect[0])
        if len(tokenizedAspect) == 1:
            if (tokenizedAspect[0] in terms):
                stopIndexofAspect = startIndexOfAspect = indices[terms.index(tokenizedAspect[0])]
        else:
            startOfAspectTerm = tokenizedAspect[0]
            indexDiff = len(tokenizedAspect) - 1
            stopOfAspectTerm = tokenizedAspect[-1]
            for ind, word in enumerate(terms):
                if startIndexOfAspect == -1 and startOfAspectTerm == word and stopOfAspectTerm == terms[
                    ind + indexDiff]:
                    startIndexOfAspect = indices[ind]
                if not startIndexOfAspect == -1 and stopOfAspectTerm == word:
                    stopIndexofAspect = indices[terms.index(word)]
                    break
    else:
        # print('aspect was implicit')
        # TODO: do it here and now moron
        startIndexOfAspect = indices[0]
        stopIndexofAspect = indices[-1]
        isImplicit = True
        # clearedAspList = list(aspects.values())
        # clearedAspList.remove(aspect)
        # for asp in clearedAspList:
        #     startIndexOfAsp = -1
        #     stopIndexofAsp = -1
        #     tokenizedAsp = nltk.word_tokenize(asp[0])
        #     if len(tokenizedAsp) == 1:
        #         if ( tokenizedAsp[0] in terms):
        #             stopIndexofAsp = startIndexOfAsp = indices[terms.index(tokenizedAsp[0])]
        #     else:
        #         startOfAspectTerm = tokenizedAsp[0]
        #         indexDiff = len(tokenizedAsp) - 1
        #         stopOfAspectTerm = tokenizedAsp[-1]
        #         for ind, word in enumerate(terms):
        #             if startIndexOfAsp == -1 and startOfAspectTerm == word and stopOfAspectTerm == terms[ind+indexDiff]:
        #                 startIndexOfAsp = indices[ind]
        #             if not startIndexOfAsp == -1 and stopOfAspectTerm == word:
        #                 stopIndexofAsp = indices[terms.index(word)]

    # except:
    #     print(tokenizedSentence)
    #     print(aspect)
    #     print(aspects)

    # if isImplicit:
    #     print(tokenizedSentence)
    #     print(aspect)
    window = list(filter(lambda x: max(0, startIndexOfAspect - windowSize) <= x[0]
                                   <= min((indices[-1]), stopIndexofAspect + windowSize) + 1 and
                                   x[1] not in nltk.word_tokenize(aspect[0]), tokenizedSentence))
    # if isImplicit:
    #     print(window)
    window = list(map(lambda x: [x[0] - startIndexOfAspect if x[0] <= startIndexOfAspect else x[0] - stopIndexofAspect,
                                 x[1], x[2], x[0]], window))
    # if isImplicit:
    #     print(window)
    #     print("===================")
    aspectList = list(aspects.values()) if type(aspects) is dict else aspects
    for asp in aspectList:
        if aspect != asp:
            for w in nltk.word_tokenize(asp[0]):
                window = list(filter(lambda x: x[1] != w, window))

    return window


def getWindowBetaBackup(tokenizedSentence, aspect, aspects, windowSize):
    indices = [t[0] for t in tokenizedSentence]
    terms = [t[1] for t in tokenizedSentence]

    startIndexOfAspect = -1
    stopIndexofAspect = -1
    tokenizedAspect = nltk.word_tokenize(aspect[0])
    if (nltk.word_tokenize(aspect[0])[0] in terms):
        startIndexOfAspect = indices[terms.index(tokenizedAspect[0])]
    if (not startIndexOfAspect == -1 and tokenizedAspect[-1] in terms[startIndexOfAspect:]):
        stopIndexofAspect = indices[terms.index(nltk.word_tokenize(aspect[0])[-1])]
    else:
        startOfAspectTerm = tokenizedAspect[0]
        stopOfAspectTerm = tokenizedAspect[-1]
        for word in terms:
            if startOfAspectTerm in word:
                startIndexOfAspect = indices[terms.index(word)]
            if not startIndexOfAspect == -1 and stopOfAspectTerm in word:
                stopIndexofAspect = indices[terms.index(word)]

    window = list(filter(lambda x: max(0, startIndexOfAspect - windowSize) <= x[0]
                                   <= min((indices[-1]), stopIndexofAspect + windowSize) + 1 and
                                   x[1] not in nltk.word_tokenize(aspect[0]), tokenizedSentence))

    # print(window)
    window = list(map(lambda x: [x[0] - startIndexOfAspect if x[0] < startIndexOfAspect else x[0] - stopIndexofAspect,
                                 x[1], x[2], x[0]], window))

    for asp in aspects:
        if aspect != asp:
            for w in nltk.word_tokenize(asp[0]):
                window = list(filter(lambda x: x[1] != w, window))

    return window


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def getFeaturesOfWindowForDeepAlpha(window, maxWindowLen, wordScoreDict):
    resultSum = np.zeros(maxWindowLen)
    resultPos = np.zeros(maxWindowLen)
    resultNeg = np.zeros(maxWindowLen)
    resultObjective = np.zeros(maxWindowLen)
    resultSumDistanceBased = np.zeros(maxWindowLen)

    midIndex = int((maxWindowLen - 1) / 2)
    if not wordScoreDict is None:
        for index, word in enumerate(window):
            if word[1] in wordScoreDict.keys():
                wordScore = wordScoreDict[word[1]]
                resultSum[midIndex + word[0]] = wordScore
                resultPos[midIndex + word[0]] = wordScore if wordScore > 0 else 0
                resultNeg[midIndex + word[0]] = wordScore if wordScore < 0 else 0
                resultObjective[midIndex + word[0]] = wordScore if -0.3 < wordScore < 0.3 else 0
                resultSumDistanceBased[midIndex + word[0]] = wordScore / (word[0] if word[0] != 0 else 1)
            else:
                synsets = list(swn.senti_synsets(word[1], get_wordnet_pos(word[2])))  # use swn
                if synsets:
                    posScore = max([synset.pos_score() for synset in synsets])
                    negScore = max([synset.neg_score() for synset in synsets])
                    objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)
                else:
                    posScore = 0
                    negScore = 0
                    objScore = 1
                wordScore = posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
                resultSum[midIndex + word[0]] = wordScore
                resultPos[midIndex + word[0]] = wordScore if wordScore > 0 else 0
                resultNeg[midIndex + word[0]] = wordScore if wordScore < 0 else 0
                resultObjective[midIndex + word[0]] = objScore if objScore > posScore and objScore > negScore else 0
                resultSumDistanceBased[midIndex + word[0]] = wordScore / (word[0] if word[0] != 0 else 1)
    else:
        for index, word in enumerate(window):
            synsets = list(swn.senti_synsets(word[1], get_wordnet_pos(word[2])))  # use swn
            if synsets:
                posScore = max([synset.pos_score() for synset in synsets])
                negScore = max([synset.neg_score() for synset in synsets])
                objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)
            else:
                posScore = 0
                negScore = 0
                objScore = 1
            wordScore = posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
            resultSum[midIndex + word[0]] = wordScore
            resultPos[midIndex + word[0]] = wordScore if wordScore > 0 else 0
            resultNeg[midIndex + word[0]] = wordScore if wordScore < 0 else 0
            resultObjective[midIndex + word[0]] = objScore if objScore > posScore and objScore > negScore else 0
            resultSumDistanceBased[midIndex + word[0]] = wordScore / (word[0] if word[0] != 0 else 1)

    return np.stack([resultSum, resultSumDistanceBased, resultPos, resultNeg, resultObjective])


def getFeaturesOfWindow(window, wordScoreDict, complWordScoreDict, objWordScoreDict=None, absolute=False,
                        useBackUp1=True, useBackUp2=True):
    scoreSum = 0
    distSum = 0
    posCount = 0
    negCount = 0
    countSum = 0
    posSUm = 0
    negSUm = 0
    maxInWindow = -1
    nearestAdjScore = 0
    nearestAdjIndex = 10000
    posPresence = [0 for _ in valuedPoses]

    ignoredWords = 0
    repairedWords = 0

    objCount = 0
    objSUm = 0
    negatedIndices = []
    if not wordScoreDict is None:
        for index, word in enumerate([w[1] for w in window]):
            if word in wordScoreDict.keys():
                wordScore = wordScoreDict[word]
                if (index > 0 and window[index - 1][1] in negations):
                    # or (index > 1 and window[index-2][1] in negations):
                    wordScore = - wordScore * 0.2
                    negatedIndices.append(index)
                scoreSum += wordScore if not absolute else abs(wordScore)
                distSum += wordScore / (window[index][0] if window[index][0] != 0 else 1)
                posCount += 1 if wordScore > 0 else 0
                posSUm += wordScore if wordScore > 0 else 0
                negCount += 1 if wordScore < 0 else 0
                negSUm += wordScore if wordScore < 0 else 0
                countSum += 1 if wordScore > 0 else -1
                maxInWindow = wordScore if abs(wordScore) > abs(maxInWindow) else maxInWindow
                # objCount += 1 if objScore > 0.5 else 0
                # objSUm += objScore if objScore > 0.5 else 0
                nearestAdjIndex = index if (
                            (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                        0] < nearestAdjIndex) else nearestAdjIndex
                if window[index][2] in valuedPoses:
                    posPresence[valuedPoses.index(window[index][2])] = 1
            elif useBackUp1 and not complWordScoreDict is None and word in complWordScoreDict.keys():
                wordScore = complWordScoreDict[word]

                if (index > 0 and window[index - 1][1] in negations):
                    # or (index > 1 and window[index-2][1] in negations):
                    wordScore = - wordScore * 0.2
                    negatedIndices.append(index)
                scoreSum += wordScore
                # distSum += wordScore/ (window[index][0] if window[index][0] != 0 else 1)
                posCount += 1 if wordScore > 0 else 0
                posSUm += wordScore if wordScore > 0 else 0
                negCount += 1 if wordScore < 0 else 0
                negSUm += wordScore if wordScore < 0 else 0
                countSum += 1 if wordScore > 0 else -1
                maxInWindow = wordScore if abs(wordScore) > abs(maxInWindow) else maxInWindow
                objCount += 1 if wordScore > 0.5 else 0
                objSUm += wordScore if wordScore < 0.4 else 0
                nearestAdjIndex = index if (
                            (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                        0] < nearestAdjIndex) else nearestAdjIndex
                if window[index][2] in valuedPoses:
                    posPresence[valuedPoses.index(window[index][2])] = 1
            elif useBackUp2:
                synsets = list(swn.senti_synsets(word, get_wordnet_pos(window[index][2])))  # use swn
                if synsets:
                    repairedWords += 1
                    posScore = max([synset.pos_score() for synset in synsets])
                    negScore = max([synset.neg_score() for synset in synsets])
                    objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)
                else:
                    posScore = 0
                    negScore = 0
                    objScore = 1
                scoreSum += posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
                wordScore = posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
                distSum += wordScore / (window[index][0] if window[index][0] != 0 else 1)
                posCount += 1 if posScore > negScore and posScore > objScore else 0
                posSUm += posScore if posScore > negScore and posScore > objScore else 0
                negCount += 1 if negScore > posScore and negScore > objScore else 0
                negSUm += negScore if negScore > posScore and negScore > objScore else 0
                countSum += 1 if posScore > negScore and posScore > objScore else -1 if negScore > objScore else 0
                maxOfPosNeg = posScore if posScore > negScore else -negScore
                maxInWindow = maxOfPosNeg if abs(maxOfPosNeg) > abs(maxInWindow) else maxInWindow
                nearestAdjIndex = index if (
                            (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                        0] < nearestAdjIndex) else nearestAdjIndex
                if window[index][2] in valuedPoses:
                    posPresence[valuedPoses.index(window[index][2])] = 1
                objCount += 1 if objScore > posScore and objScore > negScore else 0
                objSUm += objScore if objScore > posScore and objScore > negScore else 0
    else:
        for index, word in enumerate([w[1] for w in window]):
            ignoredWords += 1
            synsets = list(swn.senti_synsets(word, get_wordnet_pos(window[index][2])))  # use swn
            if synsets:
                repairedWords += 1
                posScore = max([synset.pos_score() for synset in synsets])
                negScore = max([synset.neg_score() for synset in synsets])
                objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)
            else:
                posScore = 0
                negScore = 0
                objScore = 1
            scoreSum += posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
            wordScore = posScore if posScore > negScore and posScore > objScore else -negScore if negScore > objScore else 0
            distSum += wordScore / (window[index][0] if window[index][0] != 0 else 1)
            posCount += 1 if posScore > negScore and posScore > objScore else 0
            posSUm += posScore if posScore > negScore and posScore > objScore else 0
            negCount += 1 if negScore > posScore and negScore > objScore else 0
            negSUm += negScore if negScore > posScore and negScore > objScore else 0
            countSum += 1 if posScore > negScore and posScore > objScore else -1 if negScore > objScore else 0
            maxOfPosNeg = posScore if posScore > negScore else -negScore
            maxInWindow = maxOfPosNeg if abs(maxOfPosNeg) > abs(maxInWindow) else maxInWindow
            nearestAdjIndex = index if (
                        (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                    0] < nearestAdjIndex) else nearestAdjIndex
            if window[index][2] in valuedPoses:
                posPresence[valuedPoses.index(window[index][2])] = 1
            objCount += 1 if objScore > (posScore + negScore) else 0
            objSUm += objScore if objScore > (posScore + negScore) else 0

    if nearestAdjIndex != 10000:
        if not wordScoreDict is None and window[nearestAdjIndex][1] in wordScoreDict.keys():
            nearestAdjScore = wordScoreDict[window[nearestAdjIndex][1]] if nearestAdjIndex not in negatedIndices else - \
            wordScoreDict[window[nearestAdjIndex][1]]
        else:
            synsets = list(
                swn.senti_synsets(window[nearestAdjIndex][1], get_wordnet_pos(window[nearestAdjIndex][2])))  # use swn
            if synsets:
                posScore = sum([synset.pos_score() for synset in synsets]) / len(synsets)
                negScore = sum([synset.pos_score() for synset in synsets]) / len(synsets)
                nearestAdjScore = posScore if posScore > negScore else -negScore
                nearestAdjScore = nearestAdjScore if nearestAdjIndex not in negatedIndices else -nearestAdjScore

    return ignoredWords, repairedWords, [
        scoreSum / max(1, len(window)),
        distSum,
        scoreSum, posCount, negCount, posCount / (negCount + posCount + 1), countSum, posSUm, negSUm,
        posSUm / (posSUm + negSUm) if posSUm + negSUm > 0 else 0, maxInWindow,
        nearestAdjScore,
    ]


def getFeaturesOfWindowWithBounds(window, wordScoreDict, complWordScoreDict, start=-0.3, stop=0.3):
    scoreSum = 0
    posCount = 0
    negCount = 0
    objCount = 0
    objSUm = 0
    countSum = 0
    posSUm = 0
    negSUm = 0
    maxInWindow = -1
    nearestAdjIndex = 10000
    posPresence = [0 for _ in valuedPoses]
    ignoredWords = 0
    repairedWords = 0
    nearestAdjScore = 0
    negatedIndices = []

    for index, word in enumerate([w[1] for w in window]):
        if word in wordScoreDict.keys():
            # synsets = list(swn.senti_synsets(word, get_wordnet_pos(window[index][2]))) #use swn
            # objScore = 0
            # if synsets:
            #     objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)
            wordScore = wordScoreDict[word]
            if (index > 0 and window[index - 1][1] in negations):
                # or (index > 1 and window[index-2][1] in negations):
                wordScore = - wordScore * 0.2
                negatedIndices.append(index)
            scoreSum += wordScore
            posCount += 1 if wordScore > 0 else 0
            posSUm += wordScore if wordScore > 0 else 0
            negCount += 1 if wordScore < 0 else 0
            negSUm += wordScore if wordScore < 0 else 0
            countSum += 1 if wordScore > 0 else -1
            maxInWindow = wordScore if abs(wordScore) > abs(maxInWindow) else maxInWindow
            if window[index][2] in valuedPoses:
                posPresence[valuedPoses.index(window[index][2])] = 1
            objCount += 1 if start <= wordScore <= stop else 0
            objSUm += abs(wordScore) if start <= wordScore <= stop else 0
        elif word in complWordScoreDict.keys():
            wordScore = complWordScoreDict[word]

            if (index > 0 and window[index - 1][1] in negations):
                # or (index > 1 and window[index-2][1] in negations):
                wordScore = - wordScore * 0.2
                negatedIndices.append(index)
            scoreSum += wordScore
            # distSum += wordScore/ (window[index][0] if window[index][0] != 0 else 1)
            posCount += 1 if wordScore > 0 else 0
            posSUm += wordScore if wordScore > 0 else 0
            negCount += 1 if wordScore < 0 else 0
            negSUm += wordScore if wordScore < 0 else 0
            countSum += 1 if wordScore > 0 else -1
            maxInWindow = wordScore if abs(wordScore) > abs(maxInWindow) else maxInWindow
            objCount += 1 if wordScore > 0.5 else 0
            objSUm += wordScore if wordScore < 0.4 else 0
            nearestAdjIndex = index if (
                        (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                    0] < nearestAdjIndex) else nearestAdjIndex
            if window[index][2] in valuedPoses:
                posPresence[valuedPoses.index(window[index][2])] = 1
        else:
            ignoredWords += 1
            synsets = list(swn.senti_synsets(word, get_wordnet_pos(window[index][2])))  # use swn
            if synsets:
                repairedWords += 1
                posScore = max([synset.pos_score() for synset in synsets])
                negScore = max([synset.neg_score() for synset in synsets])
                objScore = sum([synset.obj_score() for synset in synsets]) / len(synsets)

                scoreSum += posScore if posScore > negScore else -negScore
                # distSum += wordScore/ (window[index][0] if window[index][0] != 0 else 1)
                posCount += 1 if posScore > negScore else 0
                posSUm += posScore if posScore > negScore else 0
                negCount += 1 if not posScore > negScore else 0
                negSUm += negScore if not posScore > negScore else 0
                countSum += 1 if posScore > negScore else -1
                maxOfPosNeg = posScore if posScore > negScore else -negScore
                maxInWindow = maxOfPosNeg if abs(maxOfPosNeg) > abs(maxInWindow) else maxInWindow
                if window[index][2] in valuedPoses:
                    posPresence[valuedPoses.index(window[index][2])] = 1
                objCount += 1 if objScore > (posScore + negScore) else 0
                objSUm += objScore if objScore > (posScore + negScore) else 0

        nearestAdjIndex = index if (
                    (window[index][2].startswith('J') or window[index][2].startswith('R')) and window[index][
                0] < nearestAdjIndex) else nearestAdjIndex

    if nearestAdjIndex != 10000:
        if window[nearestAdjIndex][1] in wordScoreDict.keys():
            nearestAdjScore = wordScoreDict[window[nearestAdjIndex][1]] if nearestAdjIndex not in negatedIndices else - \
            wordScoreDict[window[nearestAdjIndex][1]]
        else:
            synsets = list(
                swn.senti_synsets(window[nearestAdjIndex][1], get_wordnet_pos(window[nearestAdjIndex][2])))  # use swn
            if synsets:
                posScore = sum([synset.pos_score() for synset in synsets]) / len(synsets)
                negScore = sum([synset.pos_score() for synset in synsets]) / len(synsets)
                nearestAdjScore = posScore if posScore > negScore else -negScore
                nearestAdjScore = nearestAdjScore if nearestAdjIndex not in negatedIndices else -nearestAdjScore

    return ignoredWords, repairedWords, ([
        scoreSum / max(1, len(window)), scoreSum, posCount, negCount, posCount / (negCount + posCount + 1), countSum,
        posSUm, negSUm, posSUm / (posSUm + negSUm) if posSUm + negSUm > 0 else 0, maxInWindow, nearestAdjScore,
        objCount, objSUm
    ])


def getFeaturesOfWindowForDeep(window, maxWindowLen, wordScoreDict):
    result = np.zeros(maxWindowLen)

    midIndex = int((maxWindowLen - 1) / 2)
    for index, word in enumerate(window):
        if word[1] in wordScoreDict.keys():
            wordScore = wordScoreDict[word[1]]
            result[midIndex + word[0]] = wordScore

    return result


def getTokenizedSentences(trainComments, filter=True):
    # wnl = WordNetLemmatizer()
    # ps = PorterStemmer()
    return {trainKey: getTokenizedSentence(trainComments[trainKey][0].lower(), trainComments[trainKey][1], filter) for
            trainKey in trainComments.keys()}
    # return {trainKey: nltk.word_tokenize(trainComments[trainKey][0].lower()) for trainKey in trainComments.keys()}


def getTokenizedSentence(sentence, aspectTerms, filter):
    wnl = WordNetLemmatizer()
    sentence = re.sub(MyDataParser.regex, ' ', sentence.lower())
    # ps = PorterStemmer()
    output = []
    posTag = pos_tag(nltk.word_tokenize(sentence))
    tokenizedAspects = []
    for term in aspectTerms:
        tokenizedAspects.extend(
            nltk.word_tokenize(aspectTerms[term][0]) if type(aspectTerms) is dict else nltk.word_tokenize(term[0]))
    for index, tup in enumerate(posTag):
        word = tup[0]
        pos = tup[1]
        # print(tup)

        if word in tokenizedAspects or not filter or word in negations or (
                word not in stopwords.words('english') and pos in valuedPoses):
            if len(pos) > 0:
                if word in tokenizedAspects:
                    output.append([index, word, pos])
                else:
                    output.append([index, wnl.lemmatize(word, get_wordnet_pos(pos)), pos])
            elif word in tokenizedAspects:
                output.append([index, word, pos])
            else:
                continue

    return output


def getTokenizedSentenceForDrKeshavarz(sentence):
    wnl = WordNetLemmatizer()
    # ps = PorterStemmer()
    output = []
    posTag = pos_tag(nltk.word_tokenize(sentence))
    for index, tup in enumerate(posTag):
        word = tup[0]
        pos = tup[1]
        if '@' in word or '#' in word:
            continue
        else:
            if index > 0 and '#' in posTag[index - 1][0]:
                output.append('#' + wnl.lemmatize(re.sub('[!@#$,.]', '', word.lower()), get_wordnet_pos(
                    nltk.pos_tag(re.sub('[!@#$,.]', '', word.lower()))[0][1])))
            elif index > 0 and '@' in posTag[index - 1][0]:
                output.append('@' + word)
            else:
                cleansedWord = re.sub('[!@#$,.]', '', word.lower())
                if len(pos) > 0:
                    output.append(wnl.lemmatize(cleansedWord, get_wordnet_pos(pos)))
                else:
                    output.append(word.lower())

    return output


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_bigrams(input):  # input can be either string or tokenized array
    if input is str:
        tokens = nltk.word_tokenize(input)
    else:
        tokens = input
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)

    result = [' '.join([w.lower() for w in x.split()]) for x in tokens if
              x.lower() not in stopwords.words('english') and len(x) > 1]
    return result
