# Obrada eeg signala i neuronska mreža za klasifikaciju u target i non target epohe

## Prije pokretanja potrebno je instalirati sve potrebne biblioteke
```shell
pip3 install torch numpy matplotlib scipy torchvision --user
```

## Zatim je potrebno u `data` direktorij staviti .mat datoteku u kojoj se nalaze podatci i naziv datoteke odabrati pri importu
```python
eeg_1s_raw = scio.loadmat('./data/subject_1_eeg_separated_epochs_1s.mat')
```

## Verzija pythona - testirano na 3.9 i na 3.10