# Dan 09: Rezultati Treninga

Uspešno je završeno treniranje modela kroz epohe.

### Krajnje Metrike (Epoch 100):
- **Krajnji Train Loss:** 0.0236
- **Krajnji Validation Loss:** 0.0256
- **Krajnji Train Accuracy:** 99.4%
- **Krajnji Validation Accuracy:** 100.0%

### Grafikoni performansi:
- `loss_curve.png`
- `accuracy_curve.png`


U ovom danu smo trenirali tj uporedjivali kako se model ponasa kada ima 700 i kada ima samo 100 "train_size" (na slikama je prikazan model sa 700), 
takodje smo ispratili preciznost modela kao i gubitak istog, nakon toga smo pretvorili dobijene rezultate u grafike. (PS model sa malo podataka ode u overfitting tj. accuracy bude na 100%, ali validacija tj preciznost iste ne dostize 100%, jer je model prevelik za toliko malo podataka)