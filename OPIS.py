def describe():
    print("Analizując te dane, można zauważyć: \nLiczba kolumn: W sumie mamy X kolumn w ramce danych.\n"
          "które są mało informatywne dla modelu uczenia maszynowego i mogą zostać pominięte podczas budowania modelu.\n"
          "Statystyki opisowe: Statystyki opisowe pozwalają na zrozumienie rozkładu danych numerycznych oraz wykrycie potencjalnych anomalii lub braków danych.\n"
          "Tabela przestawna: Pokazuje średnie wartości poszczególnych danych w zależności od tego czy dana osoba przeżyła czy też nie. \n"
          "Np., możemy zauważyć, czy pasażerowie z niższym wiekiem mieli większe szanse na przeżycie.")

def histogram_description():
    print("Histogram prezentuje rozkład wieku pasażerów na Titanicu. Możemy zauważyć, \n"
          "że najwięcej pasażerów koncentruje się w okolicy wieku około 20-30 lat, \n"
          "co sugeruje, że byli to głównie młodzi ludzie podróżujący na statku.")

def corr_desc():
    print("Istnieje niewielka pozytywna korelacja między wiekiem a opłatą za bilet. Oznacza to, że osoby starsze mogą mieć tendencję do płacenia wyższych opłat za bilet. \n"
          "Istnieje pewna pozytywna korelacja między liczbą rodzeństwa/małżonków a liczbą rodziców/dzieci. Może to sugerować, \n"
          "że osoby podróżujące z większą liczbą rodzeństwa/małżonków miały również tendencję do podróżowania z większą liczbą rodziców/dzieci.")

def sex_survival():
    print("Wykres słupkowy wyraźnie pokazuje, że kobiety miały znacznie większe szanse na przeżycie w porównaniu do mężczyzn na Titanicu. \n"
          "Możemy zauważyć, że średnia przeżywalność kobiet była znacznie wyższa niż średnia przeżywalność mężczyzn.")

def class_desc():
    print("Wykres słupkowy wyraźnie pokazuje, że pasażerowie podróżujący w wyższej klasie mieli znacznie większe szanse na przeżycie w porównaniu do tych podróżujących w niższych klasach na Titanicu.\n"
          "Możemy zauważyć, że średnia przeżywalność wśród pasażerów klasy pierwszej była znacznie wyższa niż wśród pasażerów klas drugiej i trzeciej.")

def age_gap_desc():
    print("Wykres słupkowy wizualizuje średnią szansę na przeżycie w różnych grupach wiekowych, podzielonych na płcie.Możemy zauważyć, \n"
          "że ogólnie rzecz biorąc, kobiety miały wyższą średnią szansę na przeżycie we wszystkich grupach wiekowych w porównaniu do mężczyzn.\n"
          "Wśród dzieci (grupa wiekowa Dziecko i Nastolatek) oraz osób w wieku średnim (grupa wiekowa Dorosły), \n"
          "różnice między szansami przeżycia dla kobiet i mężczyzn są szczególnie widoczne, gdzie kobiety miały znacznie większe szanse na przeżycie.")

