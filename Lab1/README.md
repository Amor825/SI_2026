## 🔬 Laboratorium 1 – Model Liniowy

### 1. Zmiana liczby próbek (np. 350 000)
Wpływ zwiększenia skali danych na proces treningowy i stabilność modelu.

<p align="center">
  <img src="Lab1/ss1-zad1.png" alt="Zadanie 1 - Krok 1" width="80%"><br>
  <em>Opis: Generowanie 350000 próbek danych.</em>
</p>

<p align="center">
  <img src="Lab1/ss2-zad1-2.png" alt="Zadanie 1 - Krok 2" width="80%"><br>
  <em>Opis: Inicjalizacja wag i biasów dla dużej liczby danych.</em>
</p>

<p align="center">
  <img src="Lab1/ss3-zad1-3.png" alt="Zadanie 1 - Krok 3" width="80%"><br>
  <em>Opis: Końcowy wykres / wynik funkcji straty.</em>
</p>

---

### 2. Analiza współczynnika uczenia się (`eta`)
Poniższa tabela przedstawia zachowanie modelu w zależności od dobranej wartości hiperparametru `eta`.

| Wartość `eta` | Zachowanie modelu | Zrzut ekranu (Wykres / Wynik) |
| :---: | :--- | :--- |
| **0.0001** | Bardzo powolna zbieżność modelu (niedouczenie w zadanym czasie). | <img src="Lab1/ss4-zad2-a.png" width="300"> |
| **0.001** | Nadal zbyt wolna nauka, model wymaga zbyt wielu epok. | <img src="Lab1/ss5-zad2-b.png" width="300"> |
| **0.1** | Optymalna nauka. Model szybko i stabilnie osiąga minimum. | <img src="Lab1/ss6-zad2-c.png" width="300"> |
| **1.0** | Przeuczenie / "Rozbieganie się" gradientu (za duży krok). | <img src="Lab1/ss7-zad2-d.png" width="300"> |

---

### 3. Testowanie własnej funkcji celu (Wagi i Bias)
Cel zadania: Sprawdzenie, czy model poprawnie estymuje zadane parametry dla unowocześnionej funkcji:

$$targets = 13 \cdot xs + 7 \cdot zs - 12$$

**Oczekiwane wyniki:**
* Wagi ($w$): `13` oraz `7`
* Bias ($b$): `-12`

<p align="center">
  <img src="Lab1/image.png" alt="Konfiguracja nowej funkcji celu" width="85%">
</p>

<p align="center">
  <img src="Lab1/image-1.png" alt="Wyniki optymalizacji" width="85%">
</p>

> **Wniosek:** Model pomyślnie odnalazł zdefiniowane wagi oraz bias z minimalnym błędem zaokrąglenia.
