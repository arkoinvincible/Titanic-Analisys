# Titanic-Analisys (in progress)

Opis

Projekt ten zajmuje się przewidywaniem przeżycia pasażerów na podstawie danych związanych z katastrofą Titanica. Wykorzystuje techniki uczenia maszynowego do analizy danych i prognozowania, czy dany pasażer przeżył katastrofę czy nie (przygotowywanie danych testowych w celu użycia na nich modelu w trakcie realizacji).

Pliki
train.csv: Zbiór danych treningowych zawierający informacje o pasażerach Titanica.
test.csv: Zbiór danych testowych, który zostanie użyty do generowania predykcji.

Skrypty

titanic_data.py: Skrypt Pythona zawierający kod do analizy danych, przetwarzania, uczenia maszynowego i generowania predykcji.

CLI.py: Skrypt CLI służący do wywoływania funkcji i przeglądu danych zawartych w modułach titanic_data.py, OPIS.py

OPIS.py: Skrypt Pythona zawierający opisy wykresów oraz funkcji zawartych w titanic_data.py

README.md: Niniejszy plik, zawierający opis projektu.
Zależności

Projekt wymaga zainstalowania następujących pakietów:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

### INSTRUKCJE

#### Obsługa skryptu CLI do analizy danych i predykcji przeżycia na Titanicu

1. **Uruchomienie programu**
   - Uruchom skrypt `CLI.py` w środowisku Pythona.

2. **Menu Główne**
   - Po uruchomieniu programu zostanie wyświetlone menu główne, które umożliwia wybór różnych opcji.

3. **Wybór opcji**
   - Wybierz jedną z opcji, wpisując odpowiadający jej numer.
   - Opcje są opisane poniżej:

     1. *Wczytaj dane*: Pozwala wczytać dane z pliku CSV. Podaj ścieżkę do pliku, a dane zostaną wczytane.
     2. *Zaprezentuj dane*: Wyświetla podstawowe informacje o wczytanych danych, takie jak pierwsze wiersze, statystyki opisowe i tabelę przestawną.
     3. *Generuj histogram wieku*: Generuje histogram wieku pasażerów Titanica, co pozwala zobaczyć rozkład wieku w danych.
     4. *Generuj korelację*: Generuje mapę ciepła korelacji między cechami numerycznymi w danych, co pomaga zrozumieć, które cechy są skorelowane.
     5. *Analizuj przeżycie*: Analizuje wpływ płci, klasy biletu i wieku na szanse przeżycia na Titanicu.
     6. *Oczyść i uzupełnij dane treningowe*: Oczyszcza i uzupełnia brakujące dane w ramce danych treningowych.
     7. *Przygotuj dane do modelowania*: Przygotowuje dane treningowe do modelowania, wykonując preprocessowanie.
     8. *Trenuj i optymalizuj modele*: Trenuje różne modele uczenia maszynowego i optymalizuje ich hiperparametry.
     9. *Oceń Stacking Classifier*: Ocenia wydajność Stacking Classifier na danych testowych.
     10. *Generuj predykcje na danych testowych*: Generuje predykcje przeżycia na podstawie danych testowych przy użyciu najlepszego modelu.
     11. *Wyjście*: Zamyka program.

4. **Kontynuacja działania**
   - Po wykonaniu dowolnej opcji możesz kontynuować korzystanie z programu, wybierając kolejne opcje z menu.
   - W każdej chwili możesz wybrać opcję *Wyjście*, aby zamknąć program.

Za każdym razem, gdy uruchamiasz program, pamiętaj o postępowaniu zgodnie z instrukcjami wyświetlanymi na ekranie. Jeśli napotkasz problemy lub potrzebujesz dodatkowej pomocy, skontaktuj się z odpowiednią osobą.
