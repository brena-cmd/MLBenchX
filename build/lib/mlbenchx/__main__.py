import argparse
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Mapeamento de algoritmos disponíveis
algorithm_mapping = {
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "XGBoost": XGBClassifier,
    "Dummy": DummyClassifier
}

def main():
    # Configurar argparse para lidar com entrada de múltiplos algoritmos e parâmetros
    parser = argparse.ArgumentParser(description="Benchmark de ML")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        required=True,
        help="Lista de algoritmos separados por espaço (ex: RandomForest SVM XGBoost)"
    )
    parser.add_argument(
        "--parameters",
        type=str,
        required=True,
        help="Parâmetros no formato JSON aplicáveis a cada algoritmo"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default='iris',
        help="Dataset a ser utilizado. (Ex: iris, digits, wine)"
    )
    
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Quantidade de folds para validação cruzada (padrão: 3)"
    )
    
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Quantidade de vezes para repetição do experimento (padrão: 3)"
    )
        
    args = parser.parse_args()
    n_folds = args.folds  # Quantidade de folds
    runs = args.repeat
    # Lista de algoritmos selecionados
    selected_algorithms = args.algorithms
    parameters = json.loads(args.parameters)  # Parâmetros fornecidos no formato JSON
    dataset = args.dataset
    
    # Validar os algoritmos selecionados
    for algo in selected_algorithms:
        if algo not in algorithm_mapping:
            raise ValueError(
                f"Algoritmo '{algo}' não reconhecido. Escolha entre {list(algorithm_mapping.keys())}"
            )

    # Instanciar os modelos com os parâmetros fornecidos
    list_results = []
    for run in range(1,runs+1):
        models = []
        for algo in selected_algorithms:
            AlgorithmClass = algorithm_mapping[algo]
            algo_params = parameters.get(algo, {})  # Parâmetros específicos para o algoritmo
            if algo=='XGBoost':
                if algo_params.get('tree_method') == None:
                    algo_params['tree_method']='gpu_hist'
                algo_params['use_label_encoder'] =False
                algo_params['objective']='multi:softprob'
                algo_params['eval_metric']='mlogloss'
                
            model = AlgorithmClass(**algo_params)
            models.append((algo, model))

        # Exibir os modelos instanciados
        for name, model in models:
            print(f"Modelo instanciado: {name} -> {model}")

        # Carregar o dataset de exemplo
        if dataset == "iris":
            data = load_iris()
        elif dataset == "wine":
            data = load_wine()
        elif dataset == "digits":
            data = load_digits()
            
        X, y = data.data, data.target

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()

        # Fit and transform labels to numerical values
        y = label_encoder.fit_transform(y)

        scoring = {
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted'),
            'accuracy': 'accuracy'
        }

        # Rodar o benchmark para cada modelo

        for name, model in models:
            results_ = {}
            scores = cross_validate(model, X, y, cv=n_folds, scoring=scoring)  # Validação cruzada
            results_ = {'alg':name, 'run':run}
            for metric in scoring.keys():
                results_.update({f'{metric}_mean':scores[f'test_{metric}'].mean(),
                                      f'{metric}_std':scores[f'test_{metric}'].std(),
                                      f'{metric}_var':np.var(scores[f'test_{metric}']),
                                      f'fit_time':scores['fit_time'].mean()
                                     })
            list_results.append(results_)
#             results.append(results_)
    #         results[name] = {
    #             "mean_accuracy": scores.mean(),
    #             "std_accuracy": scores.std(),
    #             "var_accuracy": np.var(scores)
    #         }

        
    print("\nResultados do Benchmark:")

    joblib.dump(list_results, f'results.pkl')
    df_results = pd.DataFrame(list_results)
    df_results.to_csv(f'results_benchmark.csv', index=False, float_format="%.4f")
    data = df_results.copy()
    for metric in ["f1_weighted_mean", "accuracy_mean", "recall_weighted_mean", "precision_weighted_mean"]:
        # Criar gráfico agrupado
        plt.figure(figsize=(10, 6))
        sns.catplot(x="run", y=metric, col="alg", kind="bar", data=data, ci=None, height=5, aspect=1)

        # Personalizar
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Média de {metric} por Run e Algoritmo")
        plt.savefig(f'barplot_{metric}.png', dpi=300, bbox_inches="tight")
        plt.clf()

    
    
    # Criar gráfico de linhas para cada métrica
    for metric in ["f1_weighted_mean", "accuracy_mean", "recall_weighted_mean", "precision_weighted_mean"]:
        plt.figure(figsize=(8, 6))
        sns.lineplot(x="run", y=metric, hue="alg", data=data, marker="o")
        plt.title(f"Evolução de {metric.capitalize()} por Repetição")
        plt.xlabel("Repetição")
        plt.ylabel(metric.capitalize())
        plt.legend(title="Algoritmo")
        plt.savefig(f'lineplot_{metric}.png', dpi=300, bbox_inches="tight")
        plt.clf()
    # Exemplo de dados com tempo
#     data["time"] = [12.3, 12.5, 12.1, 9.8, 10.1, 10.3, 8.5, 8.7, 8.6]

    # Criar gráfico de dispersão
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="fit_time", y="f1_weighted_mean", hue="alg", style="alg", data=data, s=100)
    plt.title("Tempo vs. f1_weighted_mean")
    plt.xlabel("Tempo de Execução (s)")
    plt.ylabel("f1_weighted_mean")
    plt.legend(title="Algoritmo")
    plt.savefig('scatterplot_f1.png', dpi=300, bbox_inches="tight")
    

    # Exibir os resultados
    print(df_results)

if __name__ == "__main__":
    main()
