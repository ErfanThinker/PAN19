import xml.etree.ElementTree
import csv
import re

import pickle

import  MyUtils as mu
regex = r'[-!@#$,.()\/*"]'
regexForAspects = r'[-!@#$.()\/*"]'
NumRegex = r'\d[.]?[\d]?'
patterns = [regex, NumRegex]
compiled_combined = re.compile('|'.join(x for x in patterns))

def readComments(fileName):
    print('reading comment for:', fileName)
    expCount = 0
    impCount = 0
    posCount = 0
    negCount = 0
    revCount = 0
    comments = dict()
    id = 0
    aspectLabels = {'positive': 1, 'negative': -1, 'neutral': 0, 'conflict': -2}
    with open('./datasets/liu_in_asghar_2017/' + fileName + '.txt', 'r', encoding='utf-8') as inFile:

        for line in inFile:
            if line.startswith('*') or line.startswith('[t]') or line.startswith('##'):
                continue
            elif not line.startswith('\n'): #It's a comment with aspects
                revCount +=1
                aspects, sentence = None, None
                try:
                    aspects, sentence = line.lower().split('##')
                except ValueError:
                    print(line)
                aspects = re.sub( regexForAspects, ' ', aspects).split(',')
                sentence = re.sub( regex, ' ', sentence.replace('\n', ''))
                aspectTerms = dict()
                for ind, aspect in enumerate(aspects):
                    try:
                        aspectTerm = aspect.split('[')[0]
                        rsplit = aspect.rsplit('[', 1)[1]
                        if '+' in aspect.split('[')[1]:
                            posCount +=1
                        else:
                            negCount +=1
                        if rsplit.startswith('p') or rsplit.startswith('u'):
                            impCount +=1
                        else:
                            expCount += 1
                        aspectTerms.update({fileName + '_' + str(id) + '_' + str(ind): [aspectTerm, 1 if '+' in aspect.split('[')[1] else -1, rsplit.startswith('p') or rsplit.startswith('u')]})
                    except IndexError:
                        print(fileName)
                        print(sentence)
                        print(aspect)
                        print(aspect.split('['))
                if aspectTerms:
                    comments.update({fileName + '_' + str(id): [sentence, aspectTerms]})
                id+=1
    print('dataset summary:\n Review Count:', revCount, '\nImplicit count:', impCount, '\nExplicit Count:', expCount, '\nPositive Count:', posCount, '\nNegative Count:', negCount )
    return comments

def txtToSegConvertor(fileNames):
    comments = list()
    for fileName in fileNames:
        with open('./datasets/liu_in_asghar_2017/' + fileName + '.txt', 'r', encoding='utf-8') as inFile:
            for line in inFile:
                if line.startswith('*') or line.startswith('[t]') or line.startswith('##'):
                    continue
                elif not line.startswith('\n'): #It's a comment with aspects
                    aspects, sentence = None, None
                    try:
                        aspects, sentence = line.lower().split('##')
                    except ValueError:
                        print(line)
                    aspects = re.sub( regexForAspects, ' ', aspects).split(',')
                    sentence = re.sub( regex, ' ', sentence.replace('\n', ''))

                    aspectTerms = dict()
                    for ind, aspect in enumerate(aspects):
                        try:
                            aspectTerm = aspect.split('[')[0]
                            rsplit = aspect.rsplit('[', 1)[1]
                            aspectPolarity = 2
                            if '+' in aspect.split('[')[1]:
                                aspectPolarity = 1

                            if rsplit.startswith('p') or rsplit.startswith('u'):
                                comments.append("$T$ " + re.sub('\s+', ' ', sentence).strip())
                            else:
                                comments.append(re.sub('\s+', ' ', sentence.replace(aspectTerm, "$T$", 1)).strip())

                            comments.append(aspectTerm.strip())
                            comments.append(str(aspectPolarity))

                        except IndexError:
                            print(fileName)
                            print(sentence)
                            print(aspect)
                            print(aspect.split('['))

    with open('./MemNet_ABSA-master/data/liu.seg' , 'w', encoding='utf-8') as outFile:
        for line in comments:
            outFile.write(line + "\n")

def txtToSegConvertorRaw(fileNames):
    comments = list()
    for fileName in fileNames:
        with open('./datasets/liu_in_asghar_2017/' + fileName + '.txt', 'r', encoding='utf-8') as inFile:
            for line in inFile:
                if line.startswith('*') or line.startswith('[t]') or line.startswith('##'):
                    continue
                elif not line.startswith('\n'): #It's a comment with aspects
                    aspects, sentence = None, None
                    try:
                        aspects, sentence = line.lower().split('##')
                    except ValueError:
                        print(line)
                    aspects = aspects.split(',')
                    sentence = sentence.replace('\n', '')

                    for ind, aspect in enumerate(aspects):
                        try:
                            aspectTerm = aspect.split('[')[0]
                            rsplit = aspect.rsplit('[', 1)[1]
                            aspectPolarity = 2
                            if '+' in aspect.split('[')[1]:
                                aspectPolarity = 1

                            if rsplit.startswith('p') or rsplit.startswith('u'):
                                comments.append("$T$ " + re.sub('\s+', ' ', sentence).strip())
                            else:
                                comments.append(re.sub('\s+', ' ', sentence.replace(aspectTerm, " $T$", 1)).strip())

                            comments.append(aspectTerm.strip())
                            comments.append(str(aspectPolarity))

                        except IndexError:
                            print(fileName)
                            print(sentence)
                            print(aspect)
                            print(aspect.split('['))

    with open('./MemNet_ABSA-master/data/liu.seg' , 'w', encoding='utf-8') as outFile:
        for line in comments:
            outFile.write(line + "\n")

def getSemEval2014(type= 'Laptop'):
    if (type == 'Laptop'):
        e0 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/Laptop_Train_v2.xml').getroot()
        # e1 = xml.etree.ElementTree.parse('./datasets/ABSA-SemEval2014/Laptops_Test_Data_PhaseA.xml').getroot()
        e2 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/Laptops_Test_Gold.xml').getroot()
    else: #Restaurant
        e0 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/Restaurants_Train_v2.xml').getroot()
        # e1 = xml.etree.ElementTree.parse('./datasets/ABSA-SemEval2014/Restaurants_Test_Data_PhaseA.xml').getroot()
        e2 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/Restaurants_Test_Gold.xml').getroot()

    train_comments = dict()
    test_comments = dict()
    aspectLabels = {'positive': 1, 'negative': -1, 'neutral': 0, 'conflict': -2}

    for atype in e0.findall('sentence'):
        # id = atype.get('id')
        aspectTerms = []
        text = re.sub( regex, ' ', atype.find('text').text.lower())

        if (atype.find('aspectTerms')):
            aspectTermsElement = atype.find('aspectTerms')
            for aspectTerm in aspectTermsElement.findall('aspectTerm'):
                aspectTerms.append([re.sub( regex, ' ', aspectTerm.get('term').lower()), aspectLabels[aspectTerm.get('polarity')], aspectTerm.get('from'), aspectTerm.get('to')])
        train_comments.update({ 'semEval2014' + type + str(atype.get('id')): [text, aspectTerms]})

    for atype in e2.findall('sentence'):
        # id = atype.get('id')
        aspectTerms = []
        text = re.sub(regex, ' ', atype.find('text').text.lower())
        # texts.append(atype.find('text').text)
        if (atype.find('aspectTerms')):
            aspectTermsElement = atype.find('aspectTerms')
            for aspectTerm in aspectTermsElement.findall('aspectTerm'):
                aspectTerms.append([re.sub(regex, ' ', aspectTerm.get('term').lower()), aspectLabels[aspectTerm.get('polarity')], aspectTerm.get('from'), aspectTerm.get('to')])
        test_comments.update({atype.get('id'): [text, aspectTerms]})

    return train_comments, test_comments

def getSemEval2014Trial(type= 'Laptop'):
    if (type == 'Laptop'):
        e0 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/laptops-trial.xml').getroot()
    else: #Restaurant
        e0 = xml.etree.ElementTree.parse('./ABSA-SemEval2014/restaurants-trial.xml').getroot()

    train_comments = dict()
    aspectLabels = {'positive': 1, 'negative': -1, 'neutral': 0, 'conflict': -2}

    for atype in e0.findall('sentence'):
        # id = atype.get('id')
        aspectTerms = []
        text = re.sub(regex, ' ', atype.find('text').text.lower())

        if (atype.find('aspectTerms')):
            aspectTermsElement = atype.find('aspectTerms')
            for aspectTerm in aspectTermsElement.findall('aspectTerm'):
                aspectTerms.append([re.sub(regex, ' ', aspectTerm.get('term').lower()), aspectLabels[aspectTerm.get('polarity')]])
        train_comments.update({ atype.get('id'): [text, aspectTerms]})


    return train_comments


def getLiuDataset(type= 'Computer'):
    if (type == 'Computer'):
        e0 = xml.etree.ElementTree.parse('./datasets/CustomerReviews-3-domains/Computer.xml').getroot()
    elif type == 'Router':
        e0 = xml.etree.ElementTree.parse('./datasets/CustomerReviews-3-domains/Router.xml').getroot()
    else:
        e0 = xml.etree.ElementTree.parse('./datasets/CustomerReviews-3-domains/Speaker.xml').getroot()
    aspCount = 0
    posCount = 0
    negCount = 0
    revCount = 0
    train_comments = dict()
    aspectLabels = {'positive': 1, 'negative': -1, 'neutral': 0, 'conflict': -2}

    for atype in e0.findall('sentence'):
        # id = atype.get('id')
        aspectTerms = []
        text = re.sub(regex, ' ', atype.find('text').text.lower())
        revCount +=1
        if (atype.find('aspectTerms')):
            aspectTermsElement = atype.find('aspectTerms')
            for aspectTerm in aspectTermsElement.findall('aspectTerm'):
                aspCount +=1
                if aspectLabels[aspectTerm.get('polarity')] == 1:
                    posCount +=1
                else:
                    negCount +=1
                aspectTerms.append([re.sub(regex, ' ', aspectTerm.get('term').lower()), aspectLabels[aspectTerm.get('polarity')], aspectTerm.get('from'), aspectTerm.get('to')])
        train_comments.update({ 'Liu_' + type + str(atype.get('id')): [text, aspectTerms]})

    print("Dataset: ", type)
    print('dataset summary:\n Review Count:', revCount, '\naspect count:', aspCount, '\nPositive Count:', posCount, '\nNegative Count:', negCount )
    return train_comments

def writeWordScores(fileName, wordScoreDict):
    with open(fileName + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(wordScoreDict.keys())
        spamwriter.writerow(wordScoreDict.values())

def writeWordScoresForAlgaOnFBSA(fileName, wordScoreDict):
    with open(fileName + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(wordScoreDict.keys())
        spamwriter.writerow([w[0] for w in wordScoreDict.values()])
        spamwriter.writerow([w[1] for w in wordScoreDict.values()])

def pickleWordScores(fbsaWordScores, swnScores):
    outfile = open('fbsaWordScores','wb')
    pickle.dump(fbsaWordScores,outfile)
    outfile.close()


    outfile = open('swnScores','wb')
    pickle.dump(swnScores,outfile)
    outfile.close()

def pickleData(data, fileName):
    outfile = open(fileName,'wb')
    pickle.dump(data,outfile)
    outfile.close()

def getWordScores(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        # Word_Scores_AllLabels_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_0.3__percentage__70.1686121919585.csv
        path = './RouleteWheel/RR_Word_Scores_combined_with_liu_PosNegOnly_lemmatized_filteredByPos_Laptop_window_size_5_poolSize_500_maxAge_50_nb_0.0__percentage__91.02911306702777_totalSeconds_23683.628248.csv'
        print(path)
    else :
        path = './RouleteWheel/Word_Scores_PosNegOnly_lemmatized_filteredByPos_Restaurant_window_size_5_poolSize_500_maxAge_50_nb_0.0__percentage__91.04075446278208_totalSeconds_18525.165013.csv'
        print(path)
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores
def getWordScores1(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        # Word_Scores_AllLabels_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_0.3__percentage__70.1686121919585.csv
        path = './RouleteWheel/RR1_Word_Scores_combined_with_liu_PosNegOnly_lemmatized_filteredByPos_Laptop_window_size_5_poolSize_500_maxAge_50_nb_0.0__percentage__91.02911306702777_totalSeconds_34980.291198.csv'
        print(path)
    else :
        path = './lexicon_alga/Word_Scores_AllLabels_lemmatized_filteredByPos_Restaurant_window_size_4_poolSize_500_maxAge_500_nb_0.3__percentage__80.09439200444197.csv'
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores
def getWordScores2(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        # Word_Scores_AllLabels_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_0.3__percentage__70.1686121919585.csv
        path = './RouleteWheel/RR2_Word_Scores_combined_with_liu_PosNegOnly_lemmatized_filteredByPos_Laptop_window_size_5_poolSize_500_maxAge_50_nb_0.0__percentage__91.02911306702777_totalSeconds_33952.379974.csv'
        print(path)
    else :
        path = './lexicon_alga/Word_Scores_AllLabels_lemmatized_filteredByPos_Restaurant_window_size_4_poolSize_500_maxAge_500_nb_0.3__percentage__80.09439200444197.csv'
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores

def getWordScoresExclusively(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        # Word_Scores_AllLabels_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_0.3__percentage__70.1686121919585.csv
        path = './RouleteWheel/Word_Scores_PosNegOnly_lemmatized_filteredByPos_Laptop_window_size_5_poolSize_500_maxAge_50_nb_0.0__percentage__91.04155423637344_totalSeconds_21515.137807.csv'
        print(path)
    else :
        path = './lexicon_alga/Word_Scores_AllLabels_lemmatized_filteredByPos_Restaurant_window_size_4_poolSize_500_maxAge_500_nb_0.3__percentage__80.09439200444197.csv'
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores

def getWordScoresDue(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'PosNeg':
        # Word_Scores_AllLabels_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_0.3__percentage__70.1686121919585.csv
        path = './ov*/Word_Scores_PosNeg_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_-0.3_ 0.3__percentage__99.2984349703184.csv'
    elif fileName == 'PosNeut' :
        path = './ov*/Word_Scores_PosNeut_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_-0.3_ 0.3__percentage__100.0.csv'
    else:
        path = './ov*/Word_Scores_NegNeut_lemmatized_filteredByPos_Laptop_window_size_4_poolSize_500_maxAge_250_nb_-0.3_ 0.3__percentage__100.0.csv'
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores

def readOmdCurpus(fileName):
    path = './OMD/' + fileName + '.csv'
    inFile = []
    with open(path, 'r', encoding='utf-8') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
           inFile.append(row[0:2])

    tokenziedInput = [inp[0:1] + mu.getTokenizedSentenceForDrKeshavarz(inp[1]) for inp in inFile]
    with open(fileName + '_lemmatized_' + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for row in tokenziedInput:
            spamwriter.writerow(row[0:1] + [' '.join(row[1:])])

def getFbsaWordScores(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        path = './Laptop_fbsa.csv'
    else :
        path = './Word_Scores_AllLabels_lemmatized_filteredByPos_Restaurant_window_size_3_poolSize_500_maxAge_50_nb_0.3__percentage__80.0388672959467.csv'
    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores

def getObjWordScores(fileName) :
    wordScores = dict() #words, scores
    path = ''
    if fileName == 'Laptop':
        path = './Word_Scores_Objective_stemmed_Laptop_window_size_2_poolSize_1000_maxAge_10_nb_0.0__percentage__93.12581063553826.csv'
    else :
        path = './Word_Scores_all_allWords_Objective_Restaurant_window_size_2_poolSize_1000_maxAge_1_nb_0.0__percentage__81.20488617434758.csv'

    with open(path, 'r', encoding='utf-8') as csvfile:
        wordScores = next(csv.DictReader(csvfile))
    wordScores = dict((k,float(v)) for k,v in wordScores.items()) #convert scores to float
    return wordScores

def getBingPosWords():
    words = []
    with open('./Lexicons/positive-words.txt', 'r', encoding='utf-8') as txtFile:
        for line in txtFile:
            words.append(line.lower().replace("\n",''))
    return words

def getBingNegWords():
    words = []
    with open('./Lexicons/negative-words.txt', 'r', encoding='utf-8') as txtFile:
        for line in txtFile:
            words.append(line.lower().replace("\n",''))
    return words

def getBingLiuWords():
    posWords = getBingPosWords()
    negWords = getBingNegWords()
    words = dict()
    for word in posWords:
        words.update({word: 1})
    for word in negWords:
        words.update({word: -1})
    return words

def getAfinn():
    words = dict()
    try:
        with open('./Lexicons/AFINN.txt', 'r', encoding='ISO-8859-1') as txtFile:
                for index, line in enumerate(txtFile):
                    splitIndex = str(line).rfind('\t')
                    word, score = line[:splitIndex], line[splitIndex + 1:]
                    words.update({re.sub('[!@#]', '', word.lower()): float(score.strip())})
        return words
    except ValueError:
        print('getAfinn fail')

def getMPQA():
    words = dict()
    try:
        with open('./Lexicons/MPQA.tff', 'r', encoding='ISO-8859-1') as txtFile:
                for index, line in enumerate(txtFile):
                    splitedLine = line.replace("\n",'').split(' ')
                    word, score = splitedLine[2][6:], -1 if splitedLine[5] == 'priorpolarity=negative' else 1 if splitedLine[5] == 'priorpolarity=positive' else 0
                    if splitedLine[0] == 'type=weaksubj':
                        score *= 0.5
                    words.update({re.sub('[!@#]', '', word.lower()): float(score)})
        return words
    except ValueError:
        print('getMPQA fail')

def getNRCHashTag():
    words = dict()
    try:
        with open('./Lexicons/NRCHash.txt', 'r', encoding='utf-8') as txtFile:
                for index, line in enumerate(txtFile):
                    splitIndex = str(line).rfind('\t')
                    word, score = line[:splitIndex], line[splitIndex + 1:]
                    words.update({re.sub('[!@#]', '', word.lower()): float(score.strip())})
        return words
    except ValueError:
        print('getNRCHashTag fail')

def getSWNN():
    words = dict()
    try:
        with open('./Lexicons/SWNN.txt', 'r', encoding='utf-8') as txtFile:
                for index, line in enumerate(txtFile):
                    splitIndex = str(line).rfind('\t')
                    word, score = line[:splitIndex], line[splitIndex + 1:]
                    words.update({re.sub('[!@#]', '', word.lower()): float(score.strip())})
        return words
    except ValueError:
        print('getNRCHashTag fail')

def getSWNP():
    words = dict()
    try:
        with open('./Lexicons/SWNP.txt', 'r', encoding='utf-8') as txtFile:
                for index, line in enumerate(txtFile):
                    splitIndex = str(line).rfind('\t')
                    word, score = line[:splitIndex], line[splitIndex + 1:]
                    words.update({re.sub('[!@#]', '', word.lower()): float(score.strip())})
        return words
    except ValueError:
        print('getNRCHashTag fail')

def getSenticWordNetDetailed():
    posWords = getSWNP()
    negWords = getSWNN()
    words = dict()
    for word in posWords.keys():
        words.update({word: [posWords[word], negWords[word], 1-(posWords[word] + negWords[word])]})
    return words

def getSenticWordNet():
    posWords = getSWNP()
    negWords = getSWNN()
    words = dict()
    for word in posWords.keys():
        words.update({word: (0 - negWords[word]) if negWords[word] > posWords[word] else posWords[word]})

    return words

def getSenti140Words():
    words = dict()
    try:
        with open('./Lexicons/Sentiment140.txt', 'r', encoding='utf-8') as txtFile:
                for index, line in enumerate(txtFile):
                    word, score = line.split('\t', 1)
                    words.update({re.sub('[!@#]', '', word.lower()): float(score.strip())})
        return words
    except ValueError:
        print('getSenti140Words fail')

def writePerformance(performanceOfMethods, fileName, extra= ""):
    with open(fileName + '_w2_' + extra + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        methodNames =  list(performanceOfMethods.keys())
        spamwriter.writerow(methodNames)
        for key in methodNames:
            for row in performanceOfMethods[key]:
                spamwriter.writerow(row[:4])
    with open(fileName + '_w3_' + extra + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        methodNames =  list(performanceOfMethods.keys())
        spamwriter.writerow(methodNames)
        for key in methodNames:
            for row in performanceOfMethods[key]:
                spamwriter.writerow(row[4:8])
    # with open(fileName + '_w3' + '.csv', 'w', encoding='utf-8') as csvfile:
    #     spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    #     methodNames =  list(performanceOfMethods.keys())
    #     spamwriter.writerow(methodNames)
    #     for key in methodNames:
    #         for row in performanceOfMethods[key]:
    #             spamwriter.writerow(row[4:6])
    with open(fileName + '_all_' + extra + '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        methodNames =  list(performanceOfMethods.keys())
        spamwriter.writerow(methodNames)
        for key in methodNames:
            for row in performanceOfMethods[key]:
                spamwriter.writerow(row[8:])


def writeFeatures(fileName, allTrainTests, predict, testComments):
    with open(fileName + '_train_features_'+ '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        headers =  ['id', 'aspectTerm', 'aspectPolarity'] + ['f' + str(i) for i,v in enumerate(allTrainTests[0][0])]
        spamwriter.writerow(headers[:-3])
        for row in allTrainTests[0]:
            # spamwriter.writerow(row)
            spamwriter.writerow(row[2:])

    with open(fileName + '_test_features_'+ '.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        headers =  ['id', 'aspectTerm', 'sentence', 'aspectPolarity', 'prediction'] + ['f' + str(i) for i,v in enumerate(allTrainTests[0][0])]
        spamwriter.writerow(headers)
        for ind, row in enumerate(allTrainTests[1]):
            # if not predict[ind] == row[2]:
            # spamwriter.writerow(row[:2] + [testComments[row[0]][0]]  + [row[2]] + [predict[ind]] + row[3:])
            spamwriter.writerow(row[2:])




def writeWordScoreToFreq(wordScoreStatistics):
    with open('wordStatistics.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for row in wordScoreStatistics:
            # spamwriter.writerow(row)
            spamwriter.writerow(row)


def writeLabels(allLabels):
    with open('labels.csv', 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for row in allLabels:
            spamwriter.writerow(row)
