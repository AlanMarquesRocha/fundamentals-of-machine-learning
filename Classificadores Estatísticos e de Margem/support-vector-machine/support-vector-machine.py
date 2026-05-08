"""
======================================================================================
            Máquina de Vetor de Suporte (SVM) - from scratch
        Dataset: Iris (Fisher, 1936) — embutido no próprio arquivo
     Bibliotecas externas usadas: APENAS numpy (álgebra linear) e matplotlib (plot)
      Tudo o mais (split, normalização, treino, métricas) é implementado manualmente.
=======================================================================================

==================================================================================
                       Author: Alan Marques da Rocha
==================================================================================

Formulação matemática (SVM linear de margem suave):
    minimizar  (1/2)·||w||² + C · Σ max(0, 1 − yᵢ·(w·xᵢ + b))

Resolvido por subgradiente descendente em batch.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1.  DATASET IRIS  (UCI Machine Learning Repository — domínio público)
#     Cada linha: sepal_length, sepal_width, petal_length, petal_width, classe
#     Classes: 0 = Iris-setosa | 1 = Iris-versicolor | 2 = Iris-virginica
# =============================================================================
IRIS_RAW = """5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.4,3.7,1.5,0.2,0
4.8,3.4,1.6,0.2,0
4.8,3.0,1.4,0.1,0
4.3,3.0,1.1,0.1,0
5.8,4.0,1.2,0.2,0
5.7,4.4,1.5,0.4,0
5.4,3.9,1.3,0.4,0
5.1,3.5,1.4,0.3,0
5.7,3.8,1.7,0.3,0
5.1,3.8,1.5,0.3,0
5.4,3.4,1.7,0.2,0
5.1,3.7,1.5,0.4,0
4.6,3.6,1.0,0.2,0
5.1,3.3,1.7,0.5,0
4.8,3.4,1.9,0.2,0
5.0,3.0,1.6,0.2,0
5.0,3.4,1.6,0.4,0
5.2,3.5,1.5,0.2,0
5.2,3.4,1.4,0.2,0
4.7,3.2,1.6,0.2,0
4.8,3.1,1.6,0.2,0
5.4,3.4,1.5,0.4,0
5.2,4.1,1.5,0.1,0
5.5,4.2,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.0,3.2,1.2,0.2,0
5.5,3.5,1.3,0.2,0
4.9,3.1,1.5,0.1,0
4.4,3.0,1.3,0.2,0
5.1,3.4,1.5,0.2,0
5.0,3.5,1.3,0.3,0
4.5,2.3,1.3,0.3,0
4.4,3.2,1.3,0.2,0
5.0,3.5,1.6,0.6,0
5.1,3.8,1.9,0.4,0
4.8,3.0,1.4,0.3,0
5.1,3.8,1.6,0.2,0
4.6,3.2,1.4,0.2,0
5.3,3.7,1.5,0.2,0
5.0,3.3,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,2.8,4.6,1.5,1
5.7,2.8,4.5,1.3,1
6.3,3.3,4.7,1.6,1
4.9,2.4,3.3,1.0,1
6.6,2.9,4.6,1.3,1
5.2,2.7,3.9,1.4,1
5.0,2.0,3.5,1.0,1
5.9,3.0,4.2,1.5,1
6.0,2.2,4.0,1.0,1
6.1,2.9,4.7,1.4,1
5.6,2.9,3.6,1.3,1
6.7,3.1,4.4,1.4,1
5.6,3.0,4.5,1.5,1
5.8,2.7,4.1,1.0,1
6.2,2.2,4.5,1.5,1
5.6,2.5,3.9,1.1,1
5.9,3.2,4.8,1.8,1
6.1,2.8,4.0,1.3,1
6.3,2.5,4.9,1.5,1
6.1,2.8,4.7,1.2,1
6.4,2.9,4.3,1.3,1
6.6,3.0,4.4,1.4,1
6.8,2.8,4.8,1.4,1
6.7,3.0,5.0,1.7,1
6.0,2.9,4.5,1.5,1
5.7,2.6,3.5,1.0,1
5.5,2.4,3.8,1.1,1
5.5,2.4,3.7,1.0,1
5.8,2.7,3.9,1.2,1
6.0,2.7,5.1,1.6,1
5.4,3.0,4.5,1.5,1
6.0,3.4,4.5,1.6,1
6.7,3.1,4.7,1.5,1
6.3,2.3,4.4,1.3,1
5.6,3.0,4.1,1.3,1
5.5,2.5,4.0,1.3,1
5.5,2.6,4.4,1.2,1
6.1,3.0,4.6,1.4,1
5.8,2.6,4.0,1.2,1
5.0,2.3,3.3,1.0,1
5.6,2.7,4.2,1.3,1
5.7,3.0,4.2,1.2,1
5.7,2.9,4.2,1.3,1
6.2,2.9,4.3,1.3,1
5.1,2.5,3.0,1.1,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3.0,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3.0,5.8,2.2,2
7.6,3.0,6.6,2.1,2
4.9,2.5,4.5,1.7,2
7.3,2.9,6.3,1.8,2
6.7,2.5,5.8,1.8,2
7.2,3.6,6.1,2.5,2
6.5,3.2,5.1,2.0,2
6.4,2.7,5.3,1.9,2
6.8,3.0,5.5,2.1,2
5.7,2.5,5.0,2.0,2
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3.0,5.5,1.8,2
7.7,3.8,6.7,2.2,2
7.7,2.6,6.9,2.3,2
6.0,2.2,5.0,1.5,2
6.9,3.2,5.7,2.3,2
5.6,2.8,4.9,2.0,2
7.7,2.8,6.7,2.0,2
6.3,2.7,4.9,1.8,2
6.7,3.3,5.7,2.1,2
7.2,3.2,6.0,1.8,2
6.2,2.8,4.8,1.8,2
6.1,3.0,4.9,1.8,2
6.4,2.8,5.6,2.1,2
7.2,3.0,5.8,1.6,2
7.4,2.8,6.1,1.9,2
7.9,3.8,6.4,2.0,2
6.4,2.8,5.6,2.2,2
6.3,2.8,5.1,1.5,2
6.1,2.6,5.6,1.4,2
7.7,3.0,6.1,2.3,2
6.3,3.4,5.6,2.4,2
6.4,3.1,5.5,1.8,2
6.0,3.0,4.8,1.8,2
6.9,3.1,5.4,2.1,2
6.7,3.1,5.6,2.4,2
6.9,3.1,5.1,2.3,2
5.8,2.7,5.1,1.9,2
6.8,3.2,5.9,2.3,2
6.7,3.3,5.7,2.5,2
6.7,3.0,5.2,2.3,2
6.3,2.5,5.0,1.9,2
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2"""

def carregar_iris():
    """
    Lê a string IRIS_RAW e devolve X (n, 4) e y (n, ) como arrays numpy.
    """
    X, y = [], []
    for linha in IRIS_RAW.strip().split('\n'):
        partes = linha.split(',')
        X.append([float(v) for v in partes[:4]])
        y.append(int(partes[4]))
    return np.array(X), np.array(y)

# =============================================================================
# 2.  PRÉ-PROCESSAMENTO  (split treino/teste e normalização z-score)
# =============================================================================
def split_train_test(X, y, proporcao_teste = 0.3, seed = 42):
    """ Embaralha as amostras e separa em treino/teste. """
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    corte = int(len(X) * proporcao_teste)
    teste, treino = indices[:corte], indices[corte:]

    return X[treino], X[teste], y[treino], y[teste]

def normalizar(X_train, X_teste):
    """ z-score: estatísticas calculadas só no treino e aplicadas a ambos """
    media = X_train.mean(axis = 0)
    desvio = X_train.std(axis = 0)
    desvio[desvio == 0] = 1.0 # Evita divisão por zero

    return (X_train - media) / desvio, (X_teste - media) / desvio

# =============================================================================
# 3.  IMPLEMENTAÇÃO DA SVM LINEAR DE MARGEM SUAVE
# =============================================================================
class SVM:
    """
    SVM linear treinada minimizando o hinge loss com regularização L2
    via subgradiente descendente em batch.

    Parâmetros
    ----------
    taxa_aprendizado : passo do gradiente (η)
    C                : peso do termo de erro (C grande = margem mais estreita,
                       menos tolerância a erros; C pequeno = margem mais larga)
    n_iteracoes      : número de épocas de treinamento
    """
    def __init__(self, taxa_aprendizado = 0.001, C = 1.0, n_iteracoes = 1000):
        self.lr = taxa_aprendizado
        self.C = C
        self.n_iter = n_iteracoes
        self.w = None
        self.b = None
        self.historico_perda = []

    def _hinge_loss(self, X, y):
        margens = y * (X @ self.w + self.b)

        return 0.5 * np.dot(self.w, self.w) + self.C * np.maximum(0, 1 - margens).sum()

    def fit(self, X, y):
        n_amostras, n_atributos = X.shape
        # garante rótulos em {-1, +1}
        y_ = np.where(y <= 0, -1, 1).astype(float)

        # Inicialização zero
        self.w = np.zeros(n_atributos)
        self.b = 0.0

        for _ in range(self.n_iter):
            margens = y_ * (X @ self.w + self.b)
            mask = margens < 1                      # amostras que violam a margem

            # subgradiente
            grad_w = self.w - self.C * (X[mask].T @ y_[mask])
            grad_b = -self.C * y_[mask].sum()

            # Passo
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            self.historico_perda.append(self._hinge_loss(X, y_))

        return self

    def decision_function(self, X):
        """ Distância (com sinal) ao hiperplano: w.x + b. """
        return X @ self.w + self.b

    def predict(self, X):
        """ Devolve rótulos em {-1, +1}. """
        return np.where(self.decision_function(X) >= 0, 1, -1)

# =============================================================================
# 4.  MÉTRICAS DE CLASSIFICAÇÃO  (implementadas do zero, sem sklearn)
# =============================================================================
def matriz_confusao(y_true, y_pred):
    """ Para classificação binária com rótulos em {-1, +1}. """
    vp = int(np.sum((y_pred == 1) & (y_true == 1)))
    vn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))

    return vp, vn, fp, fn

def acuracia(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precisao(y_true, y_pred):
    vp, _, fp, _ = matriz_confusao(y_true, y_pred)

    return vp / (vp + fp) if (vp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    vp, _, _, fn = matriz_confusao(y_true, y_pred)

    return vp / (vp + fn) if (vp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p, r = precisao(y_true, y_pred), recall(y_true, y_pred)

    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# =============================================================================
# 5.  VISUALIZAÇÃO DA MARGEM DE SEPARAÇÃO
# =============================================================================
def plotar_margem(X, y, modelo, titulo="SVM — Margem de separação"):
    plt.figure(figsize=(10, 7))

    # nuvem de pontos
    cores = {-1: 'tab:red', 1: 'tab:blue'}
    nomes = {-1: 'Setosa  (−1)', 1: 'Versicolor (+1)'}
    for classe in [-1, 1]:
        mask = y == classe
        plt.scatter(X[mask, 0], X[mask, 1], c=cores[classe],
                    label=nomes[classe], s=70, edgecolors='k', zorder=3)

    # grade densa para desenhar as três retas:  w·x+b = -1, 0, +1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # regiões coloridas (lado de cada classe)
    plt.contourf(xx, yy, np.sign(Z), levels=[-1.5, 0, 1.5],
                 colors=['#ffd6d6', '#d6e4ff'], alpha=0.55, zorder=1)

    # hiperplano (sólido) e margens (tracejadas)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                linestyles=['--', '-', '--'],
                colors=['gray', 'k', 'gray'],
                linewidths=[1.4, 2.2, 1.4], zorder=2)

    # destacar vetores de suporte (pontos com margem ≤ 1)
    margens = y * modelo.decision_function(X)
    sv = margens <= 1 + 1e-2
    plt.scatter(X[sv, 0], X[sv, 1], s=220, facecolors='none',
                edgecolors='lime', linewidths=2.2,
                label='Vetores de suporte', zorder=4)

    plt.xlabel('Comprimento da sépala (z-score)')
    plt.ylabel('Comprimento da pétala (z-score)')
    plt.title(titulo)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotar_convergencia(historico):
    plt.figure(figsize=(8, 4))
    plt.plot(historico, color='tab:purple')
    plt.xlabel('Iteração')
    plt.ylabel('Hinge loss + ½‖w‖²')
    plt.title('Curva de convergência do treinamento')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6.  PIPELINE PRINCIPAL
# =============================================================================
def main():
    print("=" * 62)
    print(" SVM IMPLEMENTADA DO ZERO  —  DATASET IRIS")
    print("=" * 62)

    # --- carrega e prepara os dados ------------------------------------------
    X, y = carregar_iris()

    # binário: Setosa (0) vs Versicolor (1)
    # 2 atributos para visualização: sepal_length (col 0) e petal_length (col 2)
    mask = y < 2
    X = X[mask][:, [0, 2]]
    y = y[mask]
    y = np.where(y == 0, -1, 1)  # mapeia para {-1, +1}

    print(f"\n  Amostras totais : {len(X)}")
    print(f"  Atributos       : sepal_length, petal_length")
    print(f"  Classes         : Setosa (−1)  vs  Versicolor (+1)")

    # split + normalização
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.3, seed=42)
    X_train, X_test = normalizar(X_train, X_test)
    print(f"  Treino / Teste  : {len(X_train)} / {len(X_test)} amostras")

    # --- treino --------------------------------------------------------------
    print("\n  Treinando SVM (gradiente descendente sobre hinge loss)...")
    modelo = SVM(taxa_aprendizado=0.001, C=1.0, n_iteracoes=100).fit(X_train, y_train)
    print(f"  w = {modelo.w}")
    print(f"  b = {modelo.b:.4f}")

    # --- avaliação -----------------------------------------------------------
    y_pred = modelo.predict(X_test)
    vp, vn, fp, fn = matriz_confusao(y_test, y_pred)

    print("\n" + "-" * 62)
    print(" MÉTRICAS DE CLASSIFICAÇÃO  (conjunto de teste)")
    print("-" * 62)
    print(f"  Matriz de confusão   VP={vp}  VN={vn}  FP={fp}  FN={fn}")
    print(f"  Acurácia .......... {acuracia(y_test, y_pred):.4f}")
    print(f"  Precisão .......... {precisao(y_test, y_pred):.4f}")
    print(f"  Recall   .......... {recall(y_test, y_pred):.4f}")
    print(f"  F1-score .......... {f1_score(y_test, y_pred):.4f}")
    print("-" * 62)

    # --- gráficos ------------------------------------------------------------
    plotar_margem(X_train, y_train, modelo,
                  titulo="SVM Linear  —  Iris (Setosa × Versicolor)  —  TREINO")
    plotar_margem(X_test, y_test, modelo,
                  titulo="SVM Linear  —  Iris (Setosa × Versicolor)  —  TESTE")
    plotar_convergencia(modelo.historico_perda)


if __name__ == "__main__":
    main()



















