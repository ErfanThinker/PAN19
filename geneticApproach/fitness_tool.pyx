def calc_fitness(genes, trainCommentsForAlga, neutralBoundary, negations):
    cdef float totalDifference = 0
    cdef list labelFitness = []
    cdef list trainComment
    cdef float localSum = 0
    cdef float expectedNeutralBoundary = 0
    for trainComment in trainCommentsForAlga:
        localSum = 0
        commentWords = trainComment[5:]
        for ind, word in enumerate(commentWords):
            if word[1] in genes and not word[1] in negations:
                localSum = localSum - genes[word[1]] if ind > 0 and commentWords[ind - 1] in negations else localSum + genes[word[1]] if not word in negations else localSum
        expectedNeutralBoundary = neutralBoundary * len(commentWords)

        # for objective
        # if trainComment[2] == 0:  # neutral
        #     labelFitness.append(expectedNeutralBoundary - localSum if localSum <= expectedNeutralBoundary else 0)
        # else:  # negative and positive
        #     labelFitness.append(abs(-expectedNeutralBoundary - localSum) if localSum >= -expectedNeutralBoundary else 0)

        # For pos/neg
        if trainComment[2] == 0:  # neutral
            labelFitness.append(abs(localSum) - expectedNeutralBoundary if abs(localSum) > expectedNeutralBoundary else 0)
        elif trainComment[2] == 1:  # positive
            labelFitness.append(expectedNeutralBoundary - localSum if localSum <= expectedNeutralBoundary else 0)
        else:  # negative
            labelFitness.append(abs(-expectedNeutralBoundary - localSum) if localSum >= -expectedNeutralBoundary else 0)
        #
        totalDifference += abs(labelFitness[-1])

    return totalDifference, labelFitness
