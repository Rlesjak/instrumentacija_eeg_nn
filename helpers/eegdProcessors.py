# Python datoteka s helper funckijama za obradu podataka dobivenih iz .mat datoteke
# koja sadrzi eeg podatke

import numpy as np
import random
from scipy import signal

# Izvuci evente iz importane .mat datoteke
# drugi argument je broj reda iz ALLDATA strukture (npr 0ti red je 'EEG target')
def getEventsFromRaw(raw_dict, setIndex = 0):
    np_set = raw_dict['ALLEEG'][0,setIndex]
    # ispis imena seta
    # print('SETNAME> ' ,np_set['setname'][0])
    # print('BROJ EVENTA> ', np_set['event'].size)
    keys = np.dtype(np_set['event'][0, 0]).names
    # print('HEADER> ', keys)
    
    # Slaganje liste evenata
    events = [] 
    for evt in np_set['event'][0]:
        arr = []
        for k in keys:
            arr.append(evt[k][0,0])
        events.append(arr)
    
    return events

# izvuci data iz .mat datoteke
def getEpochsDataFromRaw(raw_dict, setIndex = 0):
    np_set = raw_dict['ALLEEG'][0,setIndex]
    # ispis imena seta
    # print('SETNAME> ' ,np_set['setname'][0])
    # print('DATA SHAPE> ' ,np_set['data'].shape)
    return np_set['data']

def dataToListOfEpochs(data):
    list = []
    nOfEpochs = data.shape[2]
    for i in range(nOfEpochs):
        list.append(data[:,:,i])
    return list

def getRandomNumberOfEpochs(epochs, nofepochs, seed=None):
    epochsLen = len( epochs )

    if nofepochs > epochsLen:
        raise Exception("Broj trazenih epocha veci od broja dostupnih epocha!")
    # Ako je dan seed koristiti ga
    if seed != None:
        random.seed(seed)
    # Randomizira redosljed epocha u listi
    random.shuffle(epochs)
    # Reseedanje random generatora da ostatak sistema ne koristi odabrani seed
    random.seed()
    # Uzima prvih n epocha gjde je n broj trazenih epocha
    return epochs[0:nofepochs]

def getLabeledEpochs(raw_data, ratio, seed=None):
        target_data = getEpochsDataFromRaw(raw_data, 0)
        nonTarget_data = getEpochsDataFromRaw(raw_data, 1)

        target_epochs = dataToListOfEpochs(target_data)
        nonTarget_epochs = dataToListOfEpochs(nonTarget_data)

        lenTarget = len(target_epochs)
        lenNonTarget = len(nonTarget_epochs)

        rndedTargetEphs = []
        rndedNonTargetEphs = []

        if lenTarget/lenNonTarget > ratio:
            # Uzimam sve nonTarget podatke i koliko treba targeta
            rndedNonTargetEphs = getRandomNumberOfEpochs(nonTarget_epochs, lenNonTarget, seed)
            newLenTarget = int(lenNonTarget*ratio)
            rndedTargetEphs = getRandomNumberOfEpochs(target_epochs, newLenTarget, seed)
        elif lenTarget/lenNonTarget < ratio:
            # Uzimam sve target podatke i kolito treba nontargeta
            rndedTargetEphs = getRandomNumberOfEpochs(target_epochs, lenTarget, seed)
            newNonTargetLen = int(lenTarget*(1/ratio))
            rndedNonTargetEphs = getRandomNumberOfEpochs(nonTarget_epochs, newNonTargetLen, seed)

        labeledEpochs = []
        for eph in rndedTargetEphs:
            labeledEpochs.append((0, eph))
        for eph in rndedNonTargetEphs:
            labeledEpochs.append((1, eph))
        
        return labeledEpochs

# Downsaple transformer klasa
class DownSample(object):
    def __init__(self, ntimes):
        self.ntimes = ntimes
    
    def __call__(self, epoch):
        eph = []
        n = self.ntimes
        for i in range(epoch.shape[0]):
            sensorData = epoch[i, :]
            sensorData = sensorData[n-1::n]
            eph.append(sensorData)
        return np.array(eph)

# Butterworthov pojasnopropusni filtar, transformer klasa
class BPButter4(object):
    def __init__(self, low, high, fs):
        self.low = low
        self.high = high
        self.fs = fs

    def __call__(self, epoch):
        sos = signal.butter(4, [self.low, self.high], btype='bandpass', output='sos', fs=self.fs)
        eph = []
        for i in range(epoch.shape[0]):
            sensorData = epoch[i, :]
            filtered = signal.sosfilt(sos, sensorData)
            eph.append(filtered)
        return np.array(eph)
