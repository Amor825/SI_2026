## 🔬 Laboratorium 2 - Model Liniowy cd

Celem laboratorium było zbudowanie uproszczonego regresyjnego modelu liniowego z użyciem biblioteki **TensorFlow/Keras** w środowisku **Google Colab**. Dane wejściowe (`xs`, `zs`, `targets`) zostały załadowane z pliku `TF_dataset.npz` wygenerowanego w Lab 1.

Model oparty jest na architekturze sekwencyjnej (`tf.keras.Sequential`) z jedną warstwą wyjściową typu `Dense`. Jako że pracujemy na prostej regresji liniowej, nie stosujemy funkcji aktywacji – model pozostaje liniowy.

---

### 1. Konfiguracja środowiska i wczytanie danych

<p align="center">
  <img src="Lab2/ss1-importy-dane.png" alt="Import bibliotek i wczytanie danych" width="85%"><br>
  <em>Import bibliotek (TensorFlow 2.19.0, NumPy, Matplotlib) oraz wczytanie danych treningowych z pliku TF_dataset.npz. Rozmiar wejścia: 2, rozmiar wyjścia: 1.</em>
</p>

---

### 2. Ćwiczenie – Testowanie różnych funkcji optymalizacji i straty

Zgodnie z instrukcją laboratoryjną przetestowano kilka kombinacji optymizatorów i funkcji straty, obserwując różnice w przebiegu treningu oraz jakości dopasowania modelu.

---

#### 2a. Optymizator: `AdamW`, Funkcja straty: `mean_absolute_error`

<p align="center">
  <img src="Lab2/ss2-trening-adamw.png" alt="Trening AdamW MAE - przebieg" width="85%"><br>
  <em>Przebieg treningu (600 epok) z optymizatorem AdamW i funkcją straty MAE. Strata startuje od ~70 i stopniowo maleje.</em>
</p>

<p align="center">
  <img src="Lab2/ss3-wagi-wykres-adamw.png" alt="Wagi i wykres predykcji - AdamW MAE" width="85%"><br>
  <em>Wyuczone wagi: [12.71, -7.01], bias: -9.27. Wykres predykcji outputs vs targets – model nie zdołał w pełni zbiec do prawidłowych parametrów (oczekiwano: 13, -7, -12). AdamW z MAE wykazuje wolniejszą zbieżność na tym zbiorze.</em>
</p>

---

#### 2b. Optymizator: `SGD`, Funkcja straty: `mean_squared_error`

<p align="center">
  <img src="Lab2/ss4-trening-sgd.png" alt="Trening SGD MSE - przebieg" width="85%"><br>
  <em>Przebieg treningu (600 epok) z klasycznym gradientem prostym (SGD) i stratą MSE (L2-norm). Strata osiąga wartości rzędu 3e-10 – praktycznie zero, co świadczy o doskonałym dopasowaniu.</em>
</p>

<p align="center">
  <img src="Lab2/ss5-wagi-wykres-sgd.png" alt="Wagi i wykres predykcji - SGD MSE" width="85%"><br>
  <em>Wyuczone wagi: [13.0, -7.0], bias: -12.0 – niemal idealne odwzorowanie zadanej funkcji targets = 13·xs - 7·zs - 12. Wykres predykcji pokazuje perfekcyjnie liniową zależność.</em>
</p>

> **Wniosek:** SGD z funkcją straty MSE osiągnął najlepsze wyniki – wagi i bias zbiegły do prawidłowych wartości z minimalnym błędem numerycznym.

---

#### 2c. Optymizator: `RMSprop`, Funkcja straty: `mean_squared_error`

<p align="center">
  <img src="Lab2/ss6-trening-rmsprop.png" alt="Trening RMSprop MSE - przebieg" width="85%"><br>
  <em>Przebieg treningu (600 epok) z optymizatorem RMSprop i stratą MSE. W okolicach epoki 273 strata wynosi ~386 i maleje stopniowo – zbieżność wyraźnie wolniejsza niż SGD.</em>
</p>

<p align="center">
  <img src="Lab2/ss7-wagi-wykres-rmsprop.png" alt="Wagi i wykres predykcji - RMSprop MSE" width="85%"><br>
  <em>Wyuczone wagi: [13.00, -7.00], bias: -12.00 – model poprawnie odtworzył parametry funkcji. Pomimo wolniejszego treningu, RMSprop ostatecznie zbiega do prawidłowego rozwiązania.</em>
</p>

> **Wniosek:** RMSprop z MSE zbiega wolniej niż SGD (większa strata w połowie treningu), ale finalnie osiąga równie dobre wyniki wagowe. Różnica polega głównie na tempie uczenia się – SGD okazał się tu szybszy i bardziej stabilny.

---

### Podsumowanie porównania optymizatorów

| Optymizator | Funkcja straty | Końcowa strata | Wagi (oczekiwane: 13, -7) | Bias (oczekiwany: -12) |
| :---: | :---: | :---: | :---: | :---: |
| **AdamW** | MAE | ~40 | 12.71, -7.01 | -9.27 |
| **SGD** | MSE | ~3e-10 | 13.00, -7.00 | -12.00 |
| **RMSprop** | MSE | ~215 (ep. 300) | 13.00, -7.00 | -12.00 |

> SGD z MSE okazał się najskuteczniejszy dla tego zadania regresji liniowej, osiągając zbieżność praktycznie do zera i perfekcyjnie odtwarzając zadane parametry modelu.
