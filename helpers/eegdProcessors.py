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
    return np.swapaxes(np.swapaxes(data,0,2), 1, 2)
    # list = []
    # nOfEpochs = data.shape[2]
    # for i in range(nOfEpochs):
    #     list.append(data[:,:,i])
    # return list

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

def getLabeledEpochs(raw_data, ratio, seed=None, filterChain = []):
        # Iscitavanje 'data' polja iz .mat datoteke
        target_data = getEpochsDataFromRaw(raw_data, 0)
        nonTarget_data = getEpochsDataFromRaw(raw_data, 1)

        # Formatiranje podataka tako da budu kao lista epoha
        # iteriranje je onda po [epoh, elektroda, sample]
        # npr. ako trazimo epoh 3: epoh3 = target_epochs[2]
        # dobiti ce se dvodimenzionalni array pa recimo svi samplovi mjerenja elektrode 8: epoh3[7]
        target_epochs = dataToListOfEpochs(target_data)
        nonTarget_epochs = dataToListOfEpochs(nonTarget_data)

        # filtriranje podataka filtrima poslanim funkciji
        for filter in filterChain:
            target_epochs = filter(target_epochs)
            nonTarget_epochs = filter(nonTarget_epochs)


        ## uzimanje odabranog omjera target i non target mjerenja
        ## uzorci se iz ukupnih podataka uzimaju nasumicno
        ## moze se dati seed pa je moguce reproducirati izlanu listu epoha
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
        # Za svaku elektrodu izbrisi mjerenje svakih n mjesta
        for i in range(epoch.shape[0]):
            sensorData = epoch[i, :] # sensordata za elektrodu i
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


# transformator za normalizaciju mjerenja
# po (pdf: CNN for P300 Detec... sekcija 3.1 na stranici 436)
class Normalise(object):
    def __call__(self, epoch):
        eph = []
        for i in range(epoch.shape[0]):
            sensorData = epoch[i, :] # sensordata za elektrodu i
            avg = np.average(sensorData)
            std = np.std(sensorData)
            sensorData = (sensorData - avg) / std
            eph.append(sensorData)
        return np.array(eph)

        

# klasa eliminira outliere tako da provjerava standardnu devijaciju 
# maksimuma svih 16 mjerenja, takodjen minimuma svih 16 mjerenja
class F_RemoveOutliers(object):
    def __init__(self, tresh = 14):
        self.tresh = tresh
    
    def __call__(self, epochs):
        filtered = []
        for epoch in epochs:
            mins = []
            maxes = []
            for i in range(epoch.shape[0]):
                sensorData = epoch[i, :] # sensordata za elektrodu i
                mins.append(sensorData.min())
                maxes.append(sensorData.min())
            std_min = np.array(mins).std()
            std_max = np.array(maxes).std()
            # Ako je standardna devijacija veca od tresholda, ne dodaj u filtered array
            if std_min > self.tresh or std_max > self.tresh:
                pass
            else:
                filtered.append(epoch)
        return np.array(filtered)
