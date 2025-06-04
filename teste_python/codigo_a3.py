import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMG_FOLDER = 'static/img'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dataframes = {}  # Armazena os dataframes por sessão simples (sem login)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csv' not in request.files or request.files['csv'].filename == '':
        return render_template('mainpage.html', erro="Nenhum arquivo CSV enviado.")

    arquivo = request.files['csv']
    caminho = os.path.join(app.config['UPLOAD_FOLDER'], arquivo.filename)
    arquivo.save(caminho)

    try:
        df = pd.read_csv(caminho)
        dataframes['df'] = df  # Armazena temporariamente
        colunas = df.columns.tolist()
        return render_template('mainpage.html', colunas=colunas)
    except Exception as e:
        return render_template('mainpage.html', erro=f"Erro ao ler CSV: {e}")

@app.route('/analisar', methods=['POST'])
def analisar():
    colunas_selecionadas = request.form.getlist('colunas')
    df = dataframes.get('df')

    if df is None or df.empty or not colunas_selecionadas:
        return render_template('mainpage.html', erro="Dados ausentes ou colunas não selecionadas.")

    coluna_alvo = 'Churn'
    if coluna_alvo not in df.columns:
        return render_template('mainpage.html', erro=f"A coluna alvo '{coluna_alvo}' não existe no dataset.")

    df = df.dropna(subset=[coluna_alvo])
    X = df[colunas_selecionadas].apply(lambda col: pd.factorize(col)[0] if col.dtype == 'object' else col)
    y = df[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    modelo = grid_search.best_estimator_
    y_pred = modelo.predict(X_test)

    acuracia = round(accuracy_score(y_test, y_pred), 4)
    relatorio = classification_report(y_test, y_pred)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No Churn", "Churn"])
    plt.title("Matriz de Confusão")
    matriz_path = os.path.join(IMG_FOLDER, 'matriz_confusao.png')
    plt.savefig(matriz_path)
    plt.clf()

    plt.figure(figsize=(12, 8))
    plot_tree(modelo, feature_names=colunas_selecionadas, class_names=["No Churn", "Churn"], filled=True)
    plt.title("Árvore de Decisão")
    arvore_path = os.path.join(IMG_FOLDER, 'arvore.png')
    plt.savefig(arvore_path)
    plt.clf()

    return render_template('mainpage.html', resultado={
        'acuracia': acuracia,
        'relatorio': relatorio,
        'matriz_path': matriz_path,
        'arvore_path': arvore_path
    })

if __name__ == '__main__':
    app.run(debug=True)