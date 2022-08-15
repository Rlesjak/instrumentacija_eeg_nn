# Python datoteka s helper funckijama za obradu podataka dobivenih iz .mat datoteke
# koja sadrzi eeg podatke

import numpy as np

# Izvuci evente iz importane .mat datoteke
# drugi argument je broj reda iz ALLDATA strukture (npr 0ti red je 'EEG target')
def getEventsFromRaw(raw_dict, setIndex = 0):
    np_set = raw_dict['ALLEEG'][0,setIndex]
    # ispis imena seta
    print('SETNAME> ' ,np_set['setname'][0])
    print('BROJ EVENTA> ', np_set['event'].size)
    keys = np.dtype(np_set['event'][0, 0]).names
    print('HEADER> ', keys)
    
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
    print('SETNAME> ' ,np_set['setname'][0])
    print('DATA SHAPE> ' ,np_set['data'].shape)
    return np_set['data']