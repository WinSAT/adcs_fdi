import numpy as np
import itertools
import csv
import random

numRW = 4
datasetSize = 30 #per scenario
maxTime = 60 #seconds
timeOffset = 5

ktNominal = 0.029
ktFaultDeviation = 0.002
ktSigDig = 4

vbusNominal = 6
vbusFaultDeviation = 2
vbusFaultDeviationStep = 0.25


def generateFaultScenarioDict(numRW=numRW):
	rw = np.arange(numRW)+1
	faultDict = [[0]]
	for i in xrange(1,len(rw)+1):
		faultDict += [list(i) for i in list(itertools.combinations(rw,i))]
	return {idx:i for idx,i in enumerate(faultDict)}

def randStartTime(maxNum=maxTime, N=datasetSize,timeOffset=timeOffset):
	return np.random.randint(timeOffset, (maxNum-timeOffset), size=N)

def randDuration(startTimes, maxNum=maxTime, timeOffset=timeOffset):
	return np.array([np.random.randint(1,(maxNum+1-timeOffset)-t) for t in startTimes]) #removal of +1 allows min 1 sec duration

def randKtSeverity(ktNominal=ktNominal, ktFaultDeviation=ktFaultDeviation, N=datasetSize):
	return (2*ktFaultDeviation)*np.random.random_sample(N)+(ktNominal-ktFaultDeviation)

def randVbusSeverity(vbusNominal=vbusNominal, dev=vbusFaultDeviation, step=vbusFaultDeviationStep, N=datasetSize):
	return np.random.choice(np.arange(vbusNominal-dev,vbusNominal+dev+step,vbusFaultDeviationStep),N)

faultScenarioDict = generateFaultScenarioDict()
totalResults = np.array(["scenario","ktBinary", "vbusBinary", "ktFaultStartTime", "vbusFaultStartTime","ktFaultDuration", "vbusFaultDuration", "ktFaultSeverity", "vbusFaultSeverity"]).T
for scenario in faultScenarioDict.keys():
	if scenario == 0:
		#from IPython import embed; embed()
		totalResults = np.vstack((totalResults,np.zeros((totalResults.shape[0],datasetSize)).T))
		continue
	scenarioArr = np.array([scenario]*datasetSize)
	ktBinary, vbusBinary = np.array([random.choice([[1,1],[1,0],[0,1]]) for i in range(datasetSize)]).T
	
	vbusFaultStartTime = randStartTime()*vbusBinary
	vbusFaultDuration  = randDuration(vbusFaultStartTime)*vbusBinary
	vbusFaultSeverity  = randVbusSeverity()*vbusBinary

	ktFaultStartTime = randStartTime()*ktBinary
	ktFaultDuration  = randDuration(ktFaultStartTime)*ktBinary
	ktFaultSeverity  = np.around(randKtSeverity()*ktBinary, ktSigDig)

	vbusFaultSeverity = [vbusNominal if vbusBinary[idx]==0 else v for idx,v in enumerate(vbusFaultSeverity)]
	ktFaultSeverity = [ktNominal if ktBinary[idx]==0 else kt for idx,kt in enumerate(ktFaultSeverity)]

	#from IPython import embed; embed()
	scenarioResults = np.array([scenarioArr,ktBinary, vbusBinary, ktFaultStartTime, vbusFaultStartTime,ktFaultDuration, vbusFaultDuration, ktFaultSeverity, vbusFaultSeverity]).T
	totalResults = np.vstack((totalResults,scenarioResults))

with open('adcs_fdi_dataset.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(totalResults)

csvFile.close()

#results = np.array(list(itertools.product(*params)))