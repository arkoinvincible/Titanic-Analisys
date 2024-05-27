import pandas as pd
from pathlib import Path
from titanic_data import (load_data, present_data, generate_age_histogram, generate_correlation, analyze_survival,
                          fit_preprocessing, transform_data, prepare_data_for_modeling, define_preprocessing_steps,
                          train_and_optimize_models, evaluate_stacking_classifier, make_predictions_on_test_data)

def main():
    df = None
    preprocessor = None
    imputer = None
    best_model = None

    while True:
        print("\nMenu:")
        print("1. Wczytaj dane")
        print("2. Zaprezentuj dane")
        print("3. Generuj histogram wieku")
        print("4. Generuj korelację")
        print("5. Analizuj przeżycie")
        print("6. Oczyść i uzupełnij dane treningowe")
        print("7. Przygotuj dane do modelowania")
        print("8. Trenuj i optymalizuj modele")
        print("9. Oceń Stacking Classifier")
        print("10. Generuj predykcje na danych testowych")
        print("11. Wyjście\n")

        choice = input("Wybierz opcję: ")

        if choice == '1':
            path = input("Podaj ścieżkę do pliku danych: ")
            df_path = Path(path)
            if df_path.exists() and df_path.is_file():
                df = load_data(path)
                print("Dane zostały wczytane.")
            else:
                print("Nie znaleziono pliku.")
        elif choice == '2' and df is not None:
            present_data(df)
        elif choice == '3' and df is not None:
            generate_age_histogram(df)
        elif choice == '4' and df is not None:
            generate_correlation(df)
        elif choice == '5' and df is not None:
            analyze_survival(df)
        elif choice == '6' and df is not None:
            imputer, df = fit_preprocessing(df)
            print("Dane treningowe zostały oczyszczone i uzupełnione.")
        elif choice == '7' and df is not None:
            X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
            preprocessor = define_preprocessing_steps()
            print("Dane zostały przygotowane do modelowania.")
        elif choice == '8' and X_train is not None:
            best_model = train_and_optimize_models(X_train, y_train, preprocessor)
        elif choice == '9' and X_test is not None:
            evaluate_stacking_classifier(X_train, y_train, X_test, y_test, preprocessor)
        elif choice == '10':
            test_data_path = input("Podaj ścieżkę do danych testowych: ")
            if best_model is not None and imputer is not None and preprocessor is not None:
                test_df = load_data(test_data_path)
                test_df_transformed = transform_data(test_df, imputer)  # Użycie transform_data
                make_predictions_on_test_data(test_df_transformed, best_model, preprocessor)  # Bezpośrednie przekazanie przetworzonego DataFrame
            else:
                print("Model nie został wytrenowany, brakuje imputera lub preprocessora.")
            
        elif choice == '11':
            print("Zamykanie programu...")
            break
        else:
            print("Niepoprawna opcja lub brak danych. Spróbuj ponownie.")

if __name__ == "__main__":
    main()