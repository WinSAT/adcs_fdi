import numpy as np
import itertools
import random

faultScenario = range(16)
datasetSize = 50
numRW = 4
maxTime = 200 #seconds

ktNominal = 0.029
vbusNominal = 6

def randIntList(maxNum, N=datasetSize):
	return np.random.randint(maxNum, size=N)

def randList(maxNum, decimals=None, N=datasetSize):
	result = np.random.rand(N)*maxNum
	if decimals:
		return np.around(result, decimals=decimals)
	return result

def randDurationCalc(startTimes, maxTime=maxTime, decimals=None, N=datasetSize):
	result = np.random.rand(N)*(maxTime - startTimes)
	if decimals:
		result = np.around(result, decimals=decimals)
	return result

def durationCheck(startTimes, durationTimes, maxTime=maxTime):
	if all(i >= maxTime for i in startTimes+durationTimes):
		raise Exception("Fault Duration exceeds maxTime ({})".format(maxTime))

def generateFaultScenarioDict(numRW=numRW):
	rw = np.arange(numRW)+1
	faultDict = [[0]]
	for i in xrange(1,len(rw)+1):
		faultDict += [list(i) for i in list(itertools.combinations(rw,i))]
	return {idx:i for idx,i in enumerate(faultDict)}

faultScenarioDict = generateFaultScenarioDict()
faultScenario = randIntList(len(faultScenarioDict.keys()))
vbusBinary, ktBinary = randIntList(2,(2,datasetSize))

vbusFaultStartTime = randList(maxTime, 2) #edge case? what if all 200sec is fault? 0-199
#ktFaultStartTime = vbusFaultStartTime[:]
ktFaultStartTime = randList(maxTime, 2)

vbusFaultDuration = randDurationCalc(vbusFaultStartTime,decimals=2)
#ktFaultDuration = vbusFaultDuration[:]
ktFaultDuration = randDurationCalc(ktFaultStartTime,decimals=2)

durationCheck(vbusFaultStartTime,vbusFaultDuration)
durationCheck(ktFaultStartTime,ktFaultDuration)

from IPython import embed; embed()

results = np.array(list(itertools.product(*params)))