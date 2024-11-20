import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def carregar_dados():
    data = pd.read_csv('data/weather_forecast_data_updated.csv')
    data = data.dropna()
    data['Rain'] = data['Rain'].map({True: 1, False: 0})
    X = data.drop('Rain', axis=1)
    y = data['Rain']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def treinar_modelo(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def visualizar_arvore(clf, X):
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=X.columns, class_names=['No Rain', 'Rain'], filled=True, rounded=True, fontsize=10)
    plt.title("Árvore de Decisão - Previsão de Chuva", fontsize=16)
    plt.show()

def exibir_matriz_confusao(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Rain', 'Rain'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Previsão de Chuva")
    plt.show()

def exibir_importancia_features(clf, X):
    importancias = clf.feature_importances_
    features = X.columns
    importancia_df = pd.DataFrame({'Feature': features, 'Importância': importancias})
    importancia_df = importancia_df.sort_values(by='Importância', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Feature', data=importancia_df, palette='viridis')
    plt.title('Importância das Variáveis')
    plt.show()

def prever_novos_dados(clf):
    try:
        temperature = float(input("Digite a Temperatura (°C): "))
        humidity = float(input("Digite a Umidade (%): "))
        wind_speed = float(input("Digite a Velocidade do Vento (km/h): "))
        cloud_cover = float(input("Digite a Cobertura de Nuvens (%): "))
        pressure = float(input("Digite a Pressão Atmosférica (hPa): "))
    except ValueError:
        print("Por favor, insira valores numéricos válidos.")
        return

    input_data = {
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind_Speed': [wind_speed],
        'Cloud_Cover': [cloud_cover],
        'Pressure': [pressure]
    }
    input_df = pd.DataFrame(input_data)
    prediction = clf.predict(input_df)
    if prediction[0] == 1:
        print("\nPrevisão: Alta probabilidade de chuva.")
    else:
        print("\nPrevisão: Baixa probabilidade de chuva.")

def main():
    X_train, X_test, y_train, y_test = carregar_dados()
    clf = treinar_modelo(X_train, y_train)

    menu_options = {
        "1": lambda: visualizar_arvore(clf, X_train),
        "2": lambda: exibir_matriz_confusao(clf, X_test, y_test),
        "3": lambda: exibir_importancia_features(clf, X_train),
        "4": lambda: prever_novos_dados(clf),
        "5": lambda: print("Saindo do programa.")
    }

    while True:
        print("\nMenu:")
        print("1. Visualizar a árvore de decisão")
        print("2. Exibir a matriz de confusão")
        print("3. Exibir a importância das variáveis")
        print("4. Inserir novos dados para previsão")
        print("5. Sair")
        escolha = input("Escolha uma opção (1-5): ")

        if escolha in menu_options:
            menu_options[escolha]()
            if escolha == "5":
                break
        else:
            print("Opção inválida. Por favor, escolha uma opção entre 1 e 5.")

if __name__ == "__main__":
    main()
