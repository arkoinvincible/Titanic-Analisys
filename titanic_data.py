# Importowanie niezbędnych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from OPIS import describe, histogram_description, corr_desc, sex_survival, class_desc, age_gap_desc

train_data = pd.read_csv(r'train.csv')
test_data = pd.read_csv(r'test.csv')
df = train_data

train_data.columns
test_data.columns

def load_data(path):
    """Wczytuje dane z pliku CSV."""
    return pd.read_csv(path)

def present_data(df):
    print(df, df.head(), df.describe())
    print(df.head())
    print(df.describe())
    print(pd.pivot_table(df, index = 'Survived', values = ['Age','SibSp','Parch','Fare']))
    describe()

def generate_age_histogram(data):
    
    plt.figure(figsize=(10, 6))
    plt.hist(data['Age'].dropna(), bins=30, edgecolor='black')
    plt.title('Histogram Wieku Pasażerów Titanica')
    plt.xlabel('Wiek')
    plt.ylabel('Liczba Pasażerów')
    plt.grid(True)
    plt.show()
    histogram_description()
    
def generate_correlation(data):
    df_num = data[['Age','SibSp','Parch','Fare']]
    sns.heatmap(df_num.corr(), annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    corr_desc()
    
def analyze_survival(data):

    # Analiza wpływu płci
    survival_by_sex = data.groupby('Sex')['Survived'].mean()
    sns.barplot(x='Sex', y='Survived', data=data)
    plt.title('Survival by Gender')
    plt.show()
    print("Analiza wpływu płci na szanse przeżycia:")
    print("\nPłeć:")
    print(survival_by_sex)
    print(sex_survival())
    
    # Analiza wpływu klasy biletu
    survival_by_class = data.groupby('Pclass')['Survived'].mean()
    sns.barplot(x='Pclass', y='Survived', data=data)
    plt.title('Survival by Ticket Class')
    plt.show()
    print("\nAnaliza wpływu Klasy biletu na szanse przeżycia:")
    print("\nKlasa Biletu:")
    print(survival_by_class)
    class_desc()
    
    
    # Analiza wpływu grupy wiekowej
    bins = [0, 12, 18, 35, 60, 80]
    labels = ['Dziecko', 'Nastolatek', 'Dorosły', 'Średni Wiek', 'Starszy']
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    survival_by_age_group = data.groupby('AgeGroup')['Survived'].mean()
    print("\nGrupa Wiekowa:")
    print(survival_by_age_group)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=data, palette='pastel')
    plt.title('Średnia szansa na przeżycie w zależności od grupy wiekowej i płci')
    plt.xlabel('Grupa Wiekowa')
    plt.ylabel('Średnia szansa na przeżycie')
    plt.xticks(rotation=45)
    plt.legend(title='Płeć')
    plt.show()
    age_gap_desc()

"""Wstępne przetworzenie danych treningowych. Używamy algorytmu KNNImputer do uzupełnienia brakujących 
   wartości w kolumnie 'Age' za pomocą wartości średnich z pięciu najbliższych sąsiadów. 
   Następnie usuwamy niepotrzebne kolumny 'Cabin', 'Ticket' i 'Name' z ramki danych."""
   
def fit_preprocessing(df):
    imputer = KNNImputer(n_neighbors=5)
    df['Age'] = imputer.fit_transform(df[['Age']])
    columns_to_drop = ['Cabin', 'Ticket', 'Name']
    df.drop(columns=columns_to_drop, inplace=True)
    return imputer, df

expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

"""Przetwarzanie danych testowych w sposób analogiczny do danych treningowych. 
Usuwamy niepotrzebne kolumny i stosujemy imputer na kolumnie 'Age' za pomocą przekazanego 
obiektu imputer."""

def transform_data(df, imputer):
    # Zastosowanie imputera do kolumny 'Age'
    df['Age'] = imputer.transform(df[['Age']].values)
    
    # Usunięcie kolumn, które nie są potrzebne do modelowania
    columns_to_drop = ['Name', 'Ticket', 'Cabin']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Zwrócenie przetworzonego DataFrame, zachowując 'PassengerId'
    return df

"""dzielimy dane na zestawy treningowe i testowe, usuwając jednocześnie kolumnę 'Survived' 
ze zbioru X, który zawiera funkcje, a zbiór y zawiera kolumnę 'Survived' jako etykietę."""

def prepare_data_for_modeling(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

"""definiuje kroki preprocessingu dla danych numerycznych i kategorycznych, 
używając scalera StandardScaler dla danych numerycznych i kodowania OneHotEncoder 
dla danych kategorycznych."""

def define_preprocessing_steps():
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor


"""przeprowadzamy trening i optymalizację trzech różnych modeli 
(Random Forest, Logistic Regression, XGBoost) używając przekazanych 
danych treningowych i kroku preprocessingu. Wykorzystujemy GridSearchCV 
do optymalizacji hiperparametrów."""

def train_and_optimize_models(X_train, y_train, preprocessor):
    
    best_score = 0
    best_model = None
    
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    param_grid = {
        'RandomForest': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 5, 10]},
        'LogisticRegression': {'classifier__C': [0.1, 1, 10]},
        'XGBClassifier': {'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.01, 0.1]}
    }
    
    for name, model in models.items():
        if name in param_grid:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', model)])
            grid_search = GridSearchCV(pipeline, param_grid=param_grid[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            print(f"{name} best params: {grid_search.best_params_}")
            print(f"{name} best score: {grid_search.best_score_:.4f}")
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
        print(f"Najlepszy model: {type(best_model.named_steps['classifier']).__name__}, Dokładność: {best_score:.4f}")
        return best_model       
       
"""Stacking Classifier - łączy trzy modele w jednym meta-klasyfikatorze."""
        
def evaluate_stacking_classifier(X_train, y_train, X_test, y_test, preprocessor):
    estimators = [
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    
    stacking_classifier = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(), cv=5
    )
    
    stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('stacking_classifier', stacking_classifier)])
    stacking_pipeline.fit(X_train, y_train)
    stacking_predictions = stacking_pipeline.predict(X_test)
    
    print(f"Stacking Classifier Test Set Accuracy: {accuracy_score(y_test, stacking_predictions):.4f}")
    print(f"Stacking Classifier Test Set ROC AUC: {roc_auc_score(y_test, stacking_predictions):.4f}")

"""generowanie predykcji na danych testowych przy użyciu najlepszego modelu i preprocessingu."""

def make_predictions_on_test_data(test_df, best_model, preprocessor):
    # Kopiowanie 'PassengerId' przed wykonaniem przekształceń
    passenger_ids = test_df['PassengerId'].copy()
    
    # Usunięcie 'PassengerId' przed przekształceniem danych
    test_df_preprocessed = preprocessor.transform(test_df.drop(['PassengerId'], axis=1))
    
    # Generowanie predykcji
    predictions = best_model.predict(test_df_preprocessed)
    
    print("Predykcje dla danych testowych:")
    for passenger_id, pred in zip(passenger_ids, predictions):
        print(f"PassengerId: {passenger_id}, Survived: {pred}")

