import numpy as np
import itertools
import csv
import random
from IPython import embed

numRW = 4
datasetSize = 1000 #per scenario
maxTime = 60 #seconds
timeOffset = 5

ktNominal = 0.029
ktFaultDeviation = 0.002
ktSigDig = 4

vbusNominal = 6
vbusFaultDeviation = 2
vbusFaultDeviationStep = 0.25

def generateFaultScenarioDict(numRW=numRW, singleFaults=False):
	rw = np.arange(numRW)+1
	faultDict = [[0]]
	for i in xrange(1,len(rw)+1):
		faultDict += [list(i) for i in list(itertools.combinations(rw,i))]
		if singleFaults:
			break
	return {idx:i for idx,i in enumerate(faultDict)}

def randStartTime(maxNum=30, N=datasetSize,timeOffset=timeOffset):
	return np.random.randint(timeOffset, (maxNum-timeOffset), size=N)
	#return 5.0*np.ones(datasetSize)

def randDuration(startTimes, maxNum=maxTime, timeOffset=timeOffset):
	#return np.array([np.random.randint(1,(maxNum+1-timeOffset)-t) for t in startTimes]) #removal of +1 allows min 1 sec duration
	#return maxTime*np.ones(len(startTimes))
	return np.array([int(maxTime-t) for t in startTimes])

def randKtSeverity(ktNominal=ktNominal, ktFaultDeviation=ktFaultDeviation, N=datasetSize):
	return (2*ktFaultDeviation)*np.random.random_sample(N)+(ktNominal-ktFaultDeviation)
	#return np.ones(N)*(ktNominal+ktFaultDeviation)

def randVbusSeverity(vbusNominal=vbusNominal, dev=vbusFaultDeviation, step=vbusFaultDeviationStep, N=datasetSize):
	return np.random.choice(np.arange(vbusNominal-dev,vbusNominal+dev+step,vbusFaultDeviationStep),N)
	#return np.ones(N)*(vbusNominal+vbusFaultDeviation)

faultScenarioDict = generateFaultScenarioDict(singleFaults=True)
totalResults = np.array(["scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration", "ktSeverity", "vbusSeverity"]).T
for scenario in faultScenarioDict.keys():
	if scenario == 0:
		nomValues = np.array([[ktNominal,vbusNominal]]*datasetSize).T
		zeroScenario = np.vstack((np.zeros((totalResults.shape[0]-nomValues.shape[0],datasetSize)),nomValues)).T
		totalResults = np.vstack((totalResults,zeroScenario))
		continue
	scenarioArr = np.array([scenario]*datasetSize)
	ktBinary, vbusBinary = np.array([random.choice([[1,1],[1,0],[0,1]]) for i in range(datasetSize)]).T
	
	time = randStartTime()
	duration = randDuration(time)
	vbusFaultStartTime = time #randStartTime()*vbusBinary
	vbusFaultDuration  = duration #randDuration(vbusFaultStartTime)*vbusBinary
	vbusFaultSeverity  = randVbusSeverity()*vbusBinary

	ktFaultStartTime = time #randStartTime()*ktBinary
	ktFaultDuration  = duration #randDuration(ktFaultStartTime)*ktBinary
	ktFaultSeverity  = np.around(randKtSeverity()*ktBinary, ktSigDig)

	vbusFaultSeverity = [vbusNominal if vbusBinary[idx]==0 else v for idx,v in enumerate(vbusFaultSeverity)]
	ktFaultSeverity = [ktNominal if ktBinary[idx]==0 else kt for idx,kt in enumerate(ktFaultSeverity)]

	#from IPython import embed; embed()
	scenarioResults = np.array([scenarioArr,ktBinary, vbusBinary, ktFaultStartTime, vbusFaultStartTime,ktFaultDuration, vbusFaultDuration, ktFaultSeverity, vbusFaultSeverity]).T
	totalResults = np.vstack((totalResults,scenarioResults))

nums = np.hstack((["num"],np.arange(totalResults.shape[0]-1)))
totalResults = np.vstack((nums,totalResults.T)).T

with open("adcs_fdi_inputs_{}_constSeverity_singleFaults_randPre30Inception_remainDuration.csv".format(datasetSize), 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(totalResults)

csvFile.close()

#results = np.array(list(itertools.product(*params)))