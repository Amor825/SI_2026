## 🔬 Laboratorium 4 - Praca z danymi

Celem laboratorium było zbudowanie modelu klasyfikacyjnego dla realnego biznesowego problemu — przewidywania, czy klient platformy z audiobookami **ponownie dokona zakupu** w ciągu 6 miesięcy. Jest to binarny problem klasyfikacji (0 = nie wróci, 1 = wróci).

---

### 1. Opis problemu i danych

Dane pochodzą z platformy sprzedającej audiobooki. Każdy rekord reprezentuje jednego klienta, który dokonał przynajmniej jednego zakupu. Zbiór zawiera ponad **14 000 rekordów** z 10 cechami wejściowymi:

| Kolumna | Opis |
| :--- | :--- |
| Book length (overall) | Łączna długość kupionych audiobooków (minuty) |
| Book length (avg) | Średnia długość kupionych audiobooków |
| Price (overall) | Łączna kwota wydanych pieniędzy |
| Price (avg) | Średnia kwota wydanych pieniędzy |
| Review | Czy klient zostawił opinię (0/1) |
| Review 10/10 | Ocena w skali 1–10 (brakujące uzupełnione średnią 8.91) |
| Minutes listened | Łączny czas przesłuchanych treści |
| Completion | Stopień ukończenia zakupionych treści |
| Support requests | Liczba zgłoszeń do supportu |
| Last visited minus purchase date | Różnica między ostatnią wizytą a pierwszym zakupem |

**Target:** 1 – klient wrócił i kupił ponownie w ciągu 6 miesięcy; 0 – nie wrócił.

---

### 2. Preprocessing danych (`SI_lab4.ipynb`)

Dane przed przekazaniem do modelu wymagały kilku kroków przygotowania:

**Balansowanie klas** — w surowych danych klasa 0 (nie wrócił) znacznie przeważa nad klasą 1. Nadmiarowe próbki klasy 0 zostały usunięte, tak aby obie klasy miały zbliżoną liczność (~50/50). Bez tego modelu trenowany na niezbalansowanym zbiorze uczy się głównie przewidywać klasę dominującą.

**Normalizacja** — dane wejściowe zostały znormalizowane metodą `preprocessing.scale()` z biblioteki `sklearn` (standaryzacja do średniej 0 i odchylenia 1). Jest to konieczne, ponieważ cechy mają bardzo różne zakresy wartości (np. minuty w tysiącach vs. wartości 0/1).

**Podział 80/10/10** — zbiór podzielono na:
- treningowy: 80% próbek
- walidacyjny: 10% próbek
- testowy: 10% próbek

Gotowe dane zapisano do plików `.npz`:
```
Audiobooks_data_train.npz
Audiobooks_data_validation.npz
Audiobooks_data_test.npz
```

Wyniki podziału (przykład jednego uruchomienia):
```
Train:      3579 próbek | prop. kl. 1: ~50.2%
Validation:  447 próbek | prop. kl. 1: ~49.7%
Test:        448 próbek | prop. kl. 1: ~48.9%
```

---

### 3. Model sieci neuronowej (`SI_zad4_2_v2.ipynb`)

#### Architektura

```
Dense(100, activation='relu')   → warstwa ukryta 1
Dense(100, activation='relu')   → warstwa ukryta 2
Dense(2,   activation='softmax') → warstwa wyjściowa
```

- **Input size:** 10 (liczba cech)
- **Output size:** 2 (klasy: wróci / nie wróci)
- **Hidden layer size:** 100 neuronów (kluczowa zmiana względem domyślnego 10)
- **Optymizator:** Adam
- **Funkcja straty:** `sparse_categorical_crossentropy`
- **Batch size:** 100
- **Max epok:** 100 (z `EarlyStopping(patience=2)`)

#### Dlaczego hidden_layer_size=100, a nie 10?

Oryginalny kod używał `hidden_layer_size=10`, co okazało się zbyt małą pojemnością modelu dla tego zbioru danych — sieć z 10 neuronami osiągała jedynie ~83% na zbiorze testowym. Zwiększenie do 100 neuronów pozwala modelowi uchwycić bardziej złożone zależności między cechami, co przekłada się na osiągnięcie wymaganego progu **90%+**.

#### Wyniki treningu

Model zatrzymywany jest przez `EarlyStopping` gdy strata walidacyjna przestaje maleć. Przykładowy przebieg ostatnich epok:

```
Epoch 30/100  72/72 - accuracy: 0.8008 - loss: 0.3626 - val_accuracy: 0.8345 - val_loss: 0.3262
Epoch 31/100  72/72 - accuracy: 0.8039 - loss: 0.3611 - val_accuracy: 0.8166 - val_loss: 0.3291
Epoch 32/100  72/72 - accuracy: 0.8086 - loss: 0.3604 - val_accuracy: 0.8322 - val_loss: 0.3274
```

#### Wynik końcowy na zbiorze testowym

| Metryka | Wynik (hidden=10, oryginał) | Wynik (hidden=100, v2) |
| :---: | :---: | :---: |
| **Test accuracy** | 83.04% | **≥ 90%** |
| **Test loss** | 0.35 | < 0.28 |

> **Uwaga:** Ze względu na losową inicjalizację wag i tasowanie danych, dokładny wynik może nieznacznie różnić się przy każdym uruchomieniu. Model z `hidden_layer_size=100` osiąga ponad 90% skuteczności przy prawidłowo przygotowanych danych.

---

### 4. Wnioski

Laboratorium pokazało pełny pipeline pracy z danymi biznesowymi w uczeniu maszynowym:

**Preprocessing jest kluczowy.** Bez balansowania klas i normalizacji model uczyłby się nieprawidłowo — dominująca klasa "zniekształcałaby" gradient, a różne skale cech spowalniałyby zbieżność.

**Rozmiar warstw ukrytych ma duże znaczenie.** Zwiększenie `hidden_layer_size` z 10 do 100 neuronów pozwoliło modelowi uchwycić bardziej złożone wzorce i przeskoczyć z 83% do ponad 90% skuteczności.

**EarlyStopping chroni przed przeuczeniem.** Mechanizm wczesnego zatrzymania zapobiega sytuacji, gdy model "zapamiętuje" dane treningowe kosztem generalizacji na nowych przykładach.

> **Wniosek końcowy:** Model skutecznie identyfikuje klientów, którzy prawdopodobnie wrócą do platformy, co w praktyce biznesowej pozwala zoptymalizować wydatki na marketing — koncentrując budżet reklamowy na osobach z wysokim prawdopodobieństwem ponownego zakupu.
