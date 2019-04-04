# File: genetic.py
#    from chapter 12 of _Genetic Algorithms with Python_
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
import copy
import random
import statistics
import sys
import time
from bisect import bisect_left
from enum import Enum
from math import exp


def _generate_parent(length, geneSet, get_fitness):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    fitness = get_fitness(genes)
    return Chromosome(genes, fitness, Strategies.Create)


def _mutate(parent, geneSet, get_fitness):
    childGenes = parent.Genes[:]
    index = random.randrange(0, len(parent.Genes))
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Mutate)


def _mutate_custom(parent, custom_mutate, get_fitness):
    childGenes = copy.deepcopy(parent.Genes)
    custom_mutate(childGenes, fitness=parent.Fitness)
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Mutate)


def _crossover(parent, index, donor, dindex, parents, get_fitness, crossover, mutate,
               generate_parent):
    childGenes = crossover(parent.Genes, donor.Genes, parent.Fitness, donor.Fitness)
    if childGenes is None:
        # parent and donor are indistinguishable
        parents[dindex] = generate_parent()
        return mutate(parents[index])
    fitness = get_fitness(childGenes)
    return Chromosome(childGenes, fitness, Strategies.Crossover)


def get_best(get_fitness, targetLen, optimalFitness, geneSet, display,
             custom_mutate=None, custom_create=None, maxAge=None,
             poolSize=1, crossover=None, staticProbabilities=False,
             probs=None):
    if custom_mutate is None:
        def fnMutate(parent):
            return _mutate(parent, geneSet, get_fitness)
    else:
        def fnMutate(parent):
            return _mutate_custom(parent, custom_mutate, get_fitness)

    if custom_create is None:
        def fnGenerateParent():
            return _generate_parent(targetLen, geneSet, get_fitness)
    else:
        def fnGenerateParent():
            genes = custom_create()
            return Chromosome(genes, get_fitness(genes), Strategies.Create)

    strategyLookup = {
        Strategies.Create: lambda p0, pi0, p1, pi1, o: fnGenerateParent(),
        Strategies.Mutate: lambda p0, pi0, p1, pi1, o: fnMutate(p0),
        Strategies.Crossover: lambda p0, pi0, p1, pi1, o:
        _crossover(p0, pi0, p1, pi1, o, get_fitness, crossover, fnMutate,
                   fnGenerateParent)
    }

    usedStrategies = [strategyLookup[Strategies.Mutate]]
    if crossover is not None:
        usedStrategies.append(strategyLookup[Strategies.Crossover])
        if not staticProbabilities:
            def fnNewChild(parent0, parentIndex0, parent1, parentIndex1, parents):
                return random.choice(usedStrategies)(parent0, parentIndex0, parent1, parentIndex1, parents)
        else:
            def fnNewChild(parent0, parentIndex0, parent1, parentIndex1, parents):
                rand = random.uniform(0, probs[0] + probs[1])
                if probs[0] < probs[1]:
                    return fnMutate(parent0) if rand < probs[0] \
                        else _crossover(parent0, parentIndex0, parent1, parentIndex1, parents, get_fitness,
                                        crossover, fnMutate, fnGenerateParent)
                else:
                    return fnMutate(parent0) if rand < probs[1] \
                        else _crossover(parent0, parentIndex0, parent1, parentIndex1, parents, get_fitness,
                                        crossover, fnMutate, fnGenerateParent)
    else:
        def fnNewChild(parent0, parentIndex0, parent1, parentIndex1, parents):
            return fnMutate(parent0)
    lastPercentage = 60
    for improvement in _get_improvement(fnNewChild, fnGenerateParent,
                                        maxAge, poolSize):
        display(improvement)
        if improvement.Strategy != Strategies.Create:
            f = strategyLookup[improvement.Strategy]
            usedStrategies.append(f)
        percentage = (100 * improvement.Fitness.MatchingCount / optimalFitness.MatchingCount)
        if (percentage - lastPercentage >= 10 and percentage > 70 and percentage - lastPercentage >= 5) or \
                (percentage > 90 and percentage - lastPercentage >= 1) or percentage == 100:
            lastPercentage = percentage
            yield improvement, (100 * float(improvement.Fitness.MatchingCount) / optimalFitness.MatchingCount)
        if not optimalFitness > improvement.Fitness:
            break


def getWorstbestParents(parents):
    sumOfFitnesses = sum(float(parent.Fitness) for parent in parents)
    f1 = lambda x: x
    f2 = lambda x: float(x.Fitness) / sumOfFitnesses
    f3 = lambda x: sumOfFitnesses / float(x.Fitness)
    ratioToSumOfFitness = [f(parent) for parent in parents for f in (f1, f2)]  # gives [p0, d0, p1, d1, ...]
    bests = random.choices(ratioToSumOfFitness[::2], ratioToSumOfFitness[1::2], k=2)
    reversedRatioToSumOfFitness = [f(parent) for parent in parents for f in (f1, f3)]
    worsts = random.choices(reversedRatioToSumOfFitness[::2], reversedRatioToSumOfFitness[1::2], k=3)
    for best in bests:
        if best in worsts:
            worsts.remove(best)
    return worsts[0], bests[0], bests[1]


def _get_improvement(new_child, generate_parent, maxAge, poolSize):
    bestParent = generate_parent()
    yield bestParent
    parents = [bestParent]
    historicalFitnesses = [bestParent.Fitness]
    for _ in range(poolSize - 1):
        parent = generate_parent()
        if parent.Fitness > bestParent.Fitness:
            yield parent
            bestParent = parent
            historicalFitnesses.append(parent.Fitness)
        parents.append(parent)
    while True:
        worstParent, parent, donerParent = getWorstbestParents(parents)
        worstParentInd, parentInd, donerParentInd = parents.index(worstParent), \
                                                    parents.index(parent), parents.index(donerParent)
        child = new_child(parent, parentInd, donerParent, donerParentInd, parents)
        if child.Fitness > worstParent.Fitness:
            child = child if random.uniform(0, child.Fitness + worstParent.Fitness) <= float(
                child.Fitness) else worstParent
        else:
            child = child if random.uniform(0, child.Fitness + worstParent.Fitness) <= float(
                worstParent.Fitness) else worstParent

        if worstParent.Fitness > child.Fitness:
            if maxAge is None:
                continue
            worstParent.Age += 1
            if maxAge > worstParent.Age:
                continue
            index = bisect_left(historicalFitnesses, child.Fitness, 0,
                                len(historicalFitnesses))
            difference = len(historicalFitnesses) - index
            proportionSimilar = difference / len(historicalFitnesses)
            if random.random() < exp(-proportionSimilar):
                parents[worstParentInd] = child
                continue
            parents[worstParentInd] = bestParent
            worstParent.Age = 0
            continue
        if not child.Fitness > worstParent.Fitness:
            # same fitness
            child.Age = parent.Age + 1
            parents[worstParentInd] = child
            continue
        parents[worstParentInd] = child
        worstParent.Age = 0
        if child.Fitness > bestParent.Fitness:
            yield child
            bestParent = child
            historicalFitnesses.append(child.Fitness)


class Chromosome:
    def __init__(self, genes, fitness, strategy):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Age = 0


class Strategies(Enum):
    Create = 0,
    Mutate = 1,
    Crossover = 2


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(100):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            if i < 10 or i % 10 == 9:
                print("{} {:3.2f} {:3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(timings, mean) if i > 1 else 0))
