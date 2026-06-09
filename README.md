# Sztuczna Inteligencja 2026

**Autor:** 21255

---

## 📋 Spis treści

- [Laboratorium 1 – Model Liniowy](#-laboratorium-1--model-liniowy)
- [Laboratorium 2 – Model Liniowy cd](#-laboratorium-2--model-liniowy-cd)
- [Laboratorium 3 – Klasyfikacja](#-laboratorium-3--klasyfikacja)
- [Laboratorium 4 – Praca z danymi](#-laboratorium-4--praca-z-danymi)
- [Laboratorium 5 – LSTM, GRU - regresja](#-laboratorium-5--lstm-gru---regresja)

---

## 🔬 Laboratorium 1 – Model Liniowy

Budowa uproszczonego modelu regresji liniowej od podstaw w NumPy. Eksperymenty z liczbą próbek, współczynnikiem uczenia się `eta` oraz własną funkcją celu z zadanymi wagami i biasem.

**Technologie:** Python, NumPy, Matplotlib

**Zadana funkcja:** `targets = 13·xs + 7·zs - 12`

> Model poprawnie odnalazł zdefiniowane wagi oraz bias z minimalnym błędem zaokrąglenia.

---

## 🔬 Laboratorium 2 – Model Liniowy cd

Reimplementacja modelu liniowego z użyciem biblioteki **TensorFlow/Keras** w środowisku Google Colab. Porównanie różnych optymizatorów i funkcji straty.

**Technologie:** Python, TensorFlow 2.x, Keras, Google Colab

| Optymizator | Funkcja straty | Końcowa strata | Wynik |
| :---: | :---: | :---: | :---: |
| AdamW | MAE | ~40 | wagi częściowo zbiegły |
| **SGD** | **MSE** | **~3e-10** | **idealne dopasowanie** |
| RMSprop | MSE | ~215 (ep. 300) | poprawne wagi |

> SGD z MSE okazał się najskuteczniejszy — wagi zbiegły do wartości `[13.0, -7.0]`, bias `-12.0`.

---

## 🔬 Laboratorium 3 – Klasyfikacja

Sieć neuronowa do rozpoznawania odręcznie pisanych cyfr na zbiorze **MNIST** (70 000 obrazów 28×28 px). Architektura feedforward z dwiema ukrytymi warstwami ReLU i warstwą wyjściową Softmax.

**Technologie:** Python, TensorFlow/Keras, OpenCV, Google Colab

**Architektura:** `Flatten → Dense(50, ReLU) → Dense(50, ReLU) → Dense(10, Softmax)`

| Metryka | Wynik |
| :---: | :---: |
| Test accuracy (MNIST) | **97.00%** |
| Skuteczność na własnych obrazkach | **2/3 = 67%** |

> Cyfry 3 i 5 rozpoznane z pewnością ~100%. Cyfra 9 błędnie sklasyfikowana jako 8 (36.5%) — utrata szczegółów po przeskalowaniu do 28×28.

---

## 🔬 Laboratorium 4 – Praca z danymi

Klasyfikacja binarna na zbiorze danych o audiobookach — przewidywanie czy klient **ponownie dokona zakupu** w ciągu 6 miesięcy. Pełny pipeline: preprocessing, balansowanie klas, normalizacja, podział 80/10/10.

**Technologie:** Python, TensorFlow/Keras, scikit-learn, NumPy

**Architektura:** `Dense(100, ReLU) → Dense(100, ReLU) → Dense(2, Softmax)`

| Metryka | Wynik |
| :---: | :---: |
| Test accuracy | **84.15%** |
| Test loss | **0.35** |

> Model skutecznie identyfikuje klientów, którzy prawdopodobnie wrócą do platformy, co pozwala zoptymalizować wydatki na marketing.

---

## 🔬 Laboratorium 5 – LSTM, GRU - regresja

Rekurencyjne sieci neuronowe (RNN) z bramkami **LSTM** i **GRU** do predykcji cen akcji IBM. Dane treningowe: 2006–2016, dane testowe: 2017. Eksperymenty z architekturą, optymizatorami, funkcjami straty i mechanizmem Early Stopping.

**Technologie:** Python, TensorFlow/Keras, Pandas, scikit-learn, Matplotlib

**Najlepsza konfiguracja (GRU):**

```
Architektura: GRU × 4 warstwy (units=100, Dropout=0.2)
Optimizer:    Adam
Loss:         Huber
EarlyStopping: monitor='val_loss', patience=5
```

| Konfiguracja | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: |
| LSTM bazowy (rmsprop, MSE, units=50) | 4.00 | 3.25 | 2.04% |
| GRU bazowy (rmsprop, MSE, units=50) | 4.04 | 3.59 | 2.27% |
| LSTM zoptymalizowany (Adam, Huber, units=100) | 2.99 | 2.06 | 1.29% |
| **GRU finalny (Adam, Huber, units=100)** | **1.78** | **1.15** | **0.72%** |

> Finalny model GRU osiągnął RMSE=1.78 (cel: <2.0 ✅) i myli się średnio o mniej niż 1% wartości akcji.

---

## 🔬 Laboratorium 6 – Praca z API lokalnego modelu

*(w trakcie)*

---

## 🔬 Laboratorium 7

*(w trakcie)*
