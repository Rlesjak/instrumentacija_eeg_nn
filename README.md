# Obrada eeg signala i neuronska mreža za klasifikaciju u target i non target epohe

## File u kojem je glavni kod je `loadData.ipynb`, file sa svim funckijama za obradu podataka je `/helpers/eegdProcessors.py`

### Prije pokretanja potrebno je instalirati sve potrebne biblioteke
```shell
pip3 install torch numpy matplotlib scipy torchvision --user
```
Ako se želi trenirati model nVidia grafičkom karticom treba instalirati CUDA toolkit
https://developer.nvidia.com/cuda-toolkit

te tek onda pytorch i to po uputama s ove stranice:
https://pytorch.org/get-started/locally/

npr. ako se koristi pip i windows10
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Zatim je potrebno u `data` direktorij staviti .mat datoteku u kojoj se nalaze podatci i naziv datoteke odabrati pri importu
```python
eeg_1s_raw = scio.loadmat('./data/subject_1_eeg_separated_epochs_1s.mat')
```
Za testiranje mreze koristi se i drugi .mat file s podatcima `subject_1_eeg_separated_epochs_2s.mat`, pa ga također treba staviti u data direktorij.

### Verzija pythona - testirano na 3.9 i na 3.10

### Model
---

Model je napravljen da kao input uzima mjerenja 16 elektroda po 512 samplea mjerenja. Dakle matrica dimenzija 16x512. Funckija 2d konvolucije očekuje i treću dimenziju koja predstavlja kanale "piksela", dakle kad bi se radilo o slici u boji to bi bili R, G, B kanali (vrijednost piksela za svaku osnovnu boju), pa se dvodimenzionalna matrica pretvara u trodimenzionalnu -> 1x16x512 time se podatci organiziraju tako da svaka tocka mjerenja ima jedan kanal.

Podatci prolaze kroz 3 "sloja" konvolucije i na kraju kroz klasifikator od 2 "sloja" i jednog izlaznog "sloja" s dva neurona. Svaki neuron predstavlja jednu klasu podataka (target, non target)


### Priprema podataka
---

Podatci su izvuceni iz .mat datoteke, zatim su osi matrica zamjenjene iz 16x512xXXXX u XXXXx16x512 gdje XXXX predstavlja broj epoha. Onda je moguće svakom epohu dodjeliti mjerenja i label, tj. koji je to tip epoha (target ili non target.). Tako da onda grupa podataka izgleda ovako.

```
[
	([16x512], 0),
	([16x512], 1),
	([16x512], 0),
	([16x512], 1),
	([16x512], 1),
	([16x512], 0),
	.
	.
	.
]
```

Sada kada su podatci logično organizirani koriste se za treniranje neuronske mreže. No prije samog "ubacivanja" u mrežu obrađuju se i filtriraju po potrebi.
U ovom primjeru su podatci prvo downsapleani ako je to potrebno (da se dobije oblik 16x512), zatim je signal filtriran Butterworthovim PP filtrom 4. reda od frekvencije 0.1hz do 20hz, te je zatim signal normaliziran. Parametri filtriranja i normalizacije uzeti su iz PDFa koji je dan u zadatku.

Također se pri importiranju podataka vrši jako jednostavno eliminiranje outliera koje nije baš najbolje rješeno. Izbacuju se epohi kojima je standardna devijacina maksimuma i minimuma signala svih 16 kanala veća od neke navedene. Time se maknu epohi kojima jedan ili nekoliko sondi daju jako različita mjerenja od ostalih, što bi se moglo protumačiti da je došlo do neke smetnje kod mjerenja...

Funkcije koje obrađuju podatke odvojene su u poseban file da notebook bude koliko toliko čitljiv, `helpers/eegdProcessors.py`. Importane su u prvoj čeliji notebooka:
```python
from eegdProcessors import getLabeledEpochs, DownSample, BPButter4, F_RemoveOutliers, Normalise
```

Funkcija getLabeledEpochs() vraća podatke sa svim preprocesiranjem izvršenim nad podatcima prije filtriranja. Što znaći da vrati epohe s oznakama tipa.

Filtri su implementirani kao klase kako bi se moglo lakše mjenjati parametre. Pozivaju se kod kreiranja dataseta:
```python
training_data = eegDataset(TRAINING_target_epochs, TRAINING_nonTarget_epochs, 1/5, 'abc' transform=transforms.Compose([
    DownSample(downsapling),
    BPButter4(0.1, 20, 512/downsapling),
    Normalise()
]), filterChain=[
    F_RemoveOutliers(200)
])
train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
```
Kod kreiranja dataseta moguće je birati omjer target i non target podataka. Ovdje su sva treniranja rađena na omjeru 1/5.


Podatci iz drugog .mat filea sadrže 2s mjerenja umjesto 1s (1024 umjesto 512), pa se prije korištenja downsapleaju podatci, ali svi ostali parametri filtriranja ostaju isti, jer je s takvim podatcima istreniran model.

```python
downsapling = 2
testing_data = eegDataset(TESTING_target_epochs, TESTING_nonTarget_epochs, 1/5, 'abc', transform=transforms.Compose([
    DownSample(downsapling),
    BPButter4(0.1, 20, 512/downsapling),
    Normalise()
]), filterChain=[
    F_RemoveOutliers(200)
])
testing_dataloader = DataLoader(testing_data, batch_size=4, shuffle=True)
```

### Ideje
---

Moglo bi se istrenirati model konstantnim nasumičnim mjenjanjem parametra filtriranja pa bi se možda dobio bolji model koji je manje specifičan.
> Treniranje nasumicnim podatcima: [https://demo.rlesjak.info/instrumentacija/test-nn-1000eph-rndVar-18.8.22](https://static.etvornica.hr/instrumentacija/test-nn-1000eph-rndVar-18.8.22)
