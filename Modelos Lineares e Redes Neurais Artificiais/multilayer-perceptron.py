"""
=============================================================================
 MLP (Multi-Layer Perceptron) - Implementação do zero, sem bibliotecas de ML
=============================================================================

============================================================================
Author: Alan Marques da Rocha
============================================================================

 Objetivo: Classificar a base de dados Iris (3 classes de flores) usando uma
 rede neural feed-forward treinada por retropropagação (backpropagation)
 just for fun (apenas por diversão).

 Bibliotecas usadas: apenas 'math' e 'random' (padrão do Python).
 Não utiliza-se NumPy, scikit-learn, TensorFlow, PyTorch, etc.

 Arquitetura da rede:
     [Entrada: 4 neurônios]  ->  [Oculta: 8 neurônios]  ->  [Saída: 3 neurônios]
       (atributos da flor)        (sigmoide)                  (sigmoide + one-hot)

 Para rodar no PyCharm:
     1. Crie um novo projeto (qualquer interpretador Python 3.x serve).
     2. Cole este arquivo no projeto.
     3. Clique com o botão direito sobre ele e escolha "Run 'mlp_iris'".
=============================================================================
"""
import math
import random

# matplotlib é usado APENAS para desenhar os gráficos.
# O cálculo das métricas (matriz de confusão, ROC, AUC) é feito do zero.
import matplotlib.pyplot as plt

# =============================================================================
# 1) BASE DE DADOS IRIS (embutida no código para não depender de arquivos)
# -----------------------------------------------------------------------------
# Cada linha possui 5 valores:
#   [sepal_length, sepal_width, petal_length, petal_width, classe]
# Classe: 0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica
# Fonte: UCI Machine Learning Repository (base pública).
# =============================================================================
iris_data = [[5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0],
    [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
    [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0],
    [5.7, 4.4, 1.5, 0.4, 0], [5.4, 3.9, 1.3, 0.4, 0], [5.1, 3.5, 1.4, 0.3, 0],
    [5.7, 3.8, 1.7, 0.3, 0], [5.1, 3.8, 1.5, 0.3, 0], [5.4, 3.4, 1.7, 0.2, 0],
    [5.1, 3.7, 1.5, 0.4, 0], [4.6, 3.6, 1.0, 0.2, 0], [5.1, 3.3, 1.7, 0.5, 0],
    [4.8, 3.4, 1.9, 0.2, 0], [5.0, 3.0, 1.6, 0.2, 0], [5.0, 3.4, 1.6, 0.4, 0],
    [5.2, 3.5, 1.5, 0.2, 0], [5.2, 3.4, 1.4, 0.2, 0], [4.7, 3.2, 1.6, 0.2, 0],
    [4.8, 3.1, 1.6, 0.2, 0], [5.4, 3.4, 1.5, 0.4, 0], [5.2, 4.1, 1.5, 0.1, 0],
    [5.5, 4.2, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.2, 0], [5.0, 3.2, 1.2, 0.2, 0],
    [5.5, 3.5, 1.3, 0.2, 0], [4.9, 3.6, 1.4, 0.1, 0], [4.4, 3.0, 1.3, 0.2, 0],
    [5.1, 3.4, 1.5, 0.2, 0], [5.0, 3.5, 1.3, 0.3, 0], [4.5, 2.3, 1.3, 0.3, 0],
    [4.4, 3.2, 1.3, 0.2, 0], [5.0, 3.5, 1.6, 0.6, 0], [5.1, 3.8, 1.9, 0.4, 0],
    [4.8, 3.0, 1.4, 0.3, 0], [5.1, 3.8, 1.6, 0.2, 0], [4.6, 3.2, 1.4, 0.2, 0],
    [5.3, 3.7, 1.5, 0.2, 0], [5.0, 3.3, 1.4, 0.2, 0], [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1],
    [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1],
    [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1], [6.6, 2.9, 4.6, 1.3, 1],
    [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1],
    [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1],
    [6.7, 3.1, 4.4, 1.4, 1], [5.6, 3.0, 4.5, 1.5, 1], [5.8, 2.7, 4.1, 1.0, 1],
    [6.2, 2.2, 4.5, 1.5, 1], [5.6, 2.5, 3.9, 1.1, 1], [5.9, 3.2, 4.8, 1.8, 1],
    [6.1, 2.8, 4.0, 1.3, 1], [6.3, 2.5, 4.9, 1.5, 1], [6.1, 2.8, 4.7, 1.2, 1],
    [6.4, 2.9, 4.3, 1.3, 1], [6.6, 3.0, 4.4, 1.4, 1], [6.8, 2.8, 4.8, 1.4, 1],
    [6.7, 3.0, 5.0, 1.7, 1], [6.0, 2.9, 4.5, 1.5, 1], [5.7, 2.6, 3.5, 1.0, 1],
    [5.5, 2.4, 3.8, 1.1, 1], [5.5, 2.4, 3.7, 1.0, 1], [5.8, 2.7, 3.9, 1.2, 1],
    [6.0, 2.7, 5.1, 1.6, 1], [5.4, 3.0, 4.5, 1.5, 1], [6.0, 3.4, 4.5, 1.6, 1],
    [6.7, 3.1, 4.7, 1.5, 1], [6.3, 2.3, 4.4, 1.3, 1], [5.6, 3.0, 4.1, 1.3, 1],
    [5.5, 2.5, 4.0, 1.3, 1], [5.5, 2.6, 4.4, 1.2, 1], [6.1, 3.0, 4.6, 1.4, 1],
    [5.8, 2.6, 4.0, 1.2, 1], [5.0, 2.3, 3.3, 1.0, 1], [5.6, 2.7, 4.2, 1.3, 1],
    [5.7, 3.0, 4.2, 1.2, 1], [5.7, 2.9, 4.2, 1.3, 1], [6.2, 2.9, 4.3, 1.3, 1],
    [5.1, 2.5, 3.0, 1.1, 1], [5.7, 2.8, 4.1, 1.3, 1],
    [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2],
    [6.3, 2.9, 5.6, 1.8, 2], [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2],
    [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2], [6.7, 2.5, 5.8, 1.8, 2],
    [7.2, 3.6, 6.1, 2.5, 2], [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2],
    [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2],
    [6.4, 3.2, 5.3, 2.3, 2], [6.5, 3.0, 5.5, 1.8, 2], [7.7, 3.8, 6.7, 2.2, 2],
    [7.7, 2.6, 6.9, 2.3, 2], [6.0, 2.2, 5.0, 1.5, 2], [6.9, 3.2, 5.7, 2.3, 2],
    [5.6, 2.8, 4.9, 2.0, 2], [7.7, 2.8, 6.7, 2.0, 2], [6.3, 2.7, 4.9, 1.8, 2],
    [6.7, 3.3, 5.7, 2.1, 2], [7.2, 3.2, 6.0, 1.8, 2], [6.2, 2.8, 4.8, 1.8, 2],
    [6.1, 3.0, 4.9, 1.8, 2], [6.4, 2.8, 5.6, 2.1, 2], [7.2, 3.0, 5.8, 1.6, 2],
    [7.4, 2.8, 6.1, 1.9, 2], [7.9, 3.8, 6.4, 2.0, 2], [6.4, 2.8, 5.6, 2.2, 2],
    [6.3, 2.8, 5.1, 1.5, 2], [6.1, 2.6, 5.6, 1.4, 2], [7.7, 3.0, 6.1, 2.3, 2],
    [6.3, 3.4, 5.6, 2.4, 2], [6.4, 3.1, 5.5, 1.8, 2], [6.0, 3.0, 4.8, 1.8, 2],
    [6.9, 3.1, 5.4, 2.1, 2], [6.7, 3.1, 5.6, 2.4, 2], [6.9, 3.1, 5.1, 2.3, 2],
    [5.8, 2.7, 5.1, 1.9, 2], [6.8, 3.2, 5.9, 2.3, 2], [6.7, 3.3, 5.7, 2.5, 2],
    [6.7, 3.0, 5.2, 2.3, 2], [6.3, 2.5, 5.0, 1.9, 2], [6.5, 3.0, 5.2, 2.0, 2],
    [6.2, 3.4, 5.4, 2.3, 2], [5.9, 3.0, 5.1, 1.8, 2],
]

# =============================================================================
# 2) FUNÇÕES DE ATIVAÇÃO E AUXILIARES
# =============================================================================

def sigmoid(x):
    """
    Função de ativação Sigmoid: f(x) = 1 / (1 + e^(-x))
    Mapeia qualquer número real para o intervalo (0, 1)

    Os 'if' iniciais evitam OverflowError quando x é muito grande ou pequeno
    """
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y):
    """
    Derivada da função Sigmoid. IMPORTANTE: aqui 'y' já é o valor de sigmoid(x),
    pois usamos a propriedade f'(x) = f(x) * (1 - f(x)).
    Isso evita recalcular sigmoid e poupa processamento.
    """
    return y * (1.0 - y)

def one_hot_encode(label, num_classes):
    """
    Converte um rótulo inteiro em um vetor one-hot.
    Exemplo: label=1, num_classes=3 -> [0, 1, 0].
    Esse formato é necessário porque a saída da rede tem um neurônio por classe.
    """
    vec = [0.0] * num_classes
    vec[label] = 1.0
    return vec

def normalize_features(data):
    """
    Normalização Min-Max: transforma cada atributo para o intervalo [0, 1].
    Isso ajuda a rede a convergir mais rápido, pois evita que atributos com
    escalas diferentes "dominem" o aprendizado.
    CUIDADO: Muito sensível a outliers!!!!!!!!

    Equação: x_norm = (X - min) / (max = min)
    """
    n_features = len(data[0])
    mins = [min(row[i] for row in data) for i in range(n_features)]
    maxs = [max(row[i] for row in data) for i in range(n_features)]
    return [
        [(row[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(n_features)]
        for row in data
    ]

# =============================================================================
# 3) CLASSE MLP - A REDE NEURAL EM SI
# =============================================================================
class MLP:
    """
    Multi-Layer Perceptron com 1 camada oculta.

    Estrutura:
        Entrada (n_input) --[W1, b1]--> Oculta (n_hidden) --[W2, b2]--> Saída (n_output)

    Cada conexão tem um peso (W) e cada neurônio tem um viés/bias (b).
    """

    def __init__(self, n_input, n_hidden, n_output, learning_rate = 0.1):
        # Quantidade de neurônios em cada camada:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Taxa de aprendizado: controla o "tamanho do passo" na atualização dos pesos
        # Valores comuns: 0.01, 0.05, 0.1, 0.5
        self.lr = learning_rate

        # ---------------------------------------------------------------------
        # Inicialização dos pesos com o método de XAVIER/GLOROT.
        #
        # Por que não usar simplesmente uniform(-1, 1)?
        # Pesos grandes fazem a entrada da sigmoide cair em regiões saturadas
        # (perto de 0 ou 1), onde a derivada é quase zero -> gradiente some
        # -> a rede "trava" prevendo só uma classe (acurácia ~33% em 3 classes).
        #
        # GLOROT resolve isso escolhendo o intervalo em função do número de
        # conexões: limite = sqrt(6 / (n_entradas + n_saidas))
        # Os vieses (bias) começam em ZERO, prática padrão.
        # ---------------------------------------------------------------------

        limite1 = math.sqrt(6.0 / (n_input + n_hidden))     # para a 1ª camada
        limite2 = math.sqrt(6.0 / (n_input + n_output))     # Para a 2ª camada

        # W1[i][j] = peso da entrada j para o neurônio oculto i.
        self.W1 = [
            [random.uniform(-limite1, limite2) for _ in range(n_input)]
            for _ in range(n_hidden)
        ]
        # b1[i] = bias do neurônio oculto i
        self.b1 = [0.0 for _ in range(n_hidden)]

        # W2[i][j] = peso do neurônio oculto j para o neurônio de saída i
        self.W2 = [
            [random.uniform(-limite2, limite2) for _ in range(n_hidden)]
            for _ in range(n_output)
        ]
        # b2[i] = bias do neurônio de saída i
        self.b2 = [0.0 for _ in range(n_output)]

    # -------------------------------------------------------------------------
    # FORWARD PROPAGATION (Propagação para frente)
    # -------------------------------------------------------------------------
    def foward(self, x):
        """
        Recebe um vetor de entrada 'x' e calcula a saída da rede.
        Para cada neurônio z = soma(peso * entrada) + bias ;  a = sigmoid(z)
        """
        # ---- Camada oculta ----
        self.hidden_output = []
        for i in range(self.n_hidden):
            # Soma ponderada das entradas + bias
            z = sum(self.W1[i][j] * x[j] for j in range(self.n_input)) + self.b1[i]
            # Aplica a função de ativação
            self.hidden_output.append(sigmoid(z))

        # ---- Camada de saída ----
        self.output = []
        for i in range(self.n_output):
            z = sum(self.W2[i][j] * self.hidden_output[j]
                    for j in range(self.n_hidden)) + self.b2[i]
            self.output.append(sigmoid(z))

        return self.output

    # -------------------------------------------------------------------------
    # BACKWARD PROPAGATION (Retropropagação do erro)
    # -------------------------------------------------------------------------
    def backward(self, x, y_true):
        """
        Compara a saída obtida com a esperada (y_true) e ajusta os pesos
        usando a Regra Delta (gradiente descendente).
        """

        # 1) Erro e delta da camada de SAÍDA
        #    erro = (esperado - obtido)
        #    delta = erro * derivada_da_ativacao
        output_deltas = []
        for i in range(self.n_output):
            erro = y_true[i] - self.output[i]
            delta = erro * sigmoid_derivative(self.output[i])
            output_deltas.append(delta)

        # 2) Erro e delta da camada OCULTA
        #    O erro de cada neurônio oculto é a soma ponderada dos deltas
        #    da camada seguinte (é por isso que se chama "retro"propagação).
        hidden_deltas = []
        for j in range(self.n_hidden):
            erro = sum(output_deltas[i] * self.W2[i][j] for i in range(self.n_output))
            delta = erro * sigmoid_derivative(self.hidden_output[j])
            hidden_deltas.append(delta)

        # 3) Atualiza os pesos W2 e biases b2 (entre oculta e saída)
        for i in range(self.n_output):
            for j in range(self.n_hidden):
                # Novo peso  = peso atual + (taxa * delta * entrada_do_neuronio)
                self.W2[i][j] += self.lr * output_deltas[i] * self.hidden_output[j]
            self.b2[i] += self.lr * output_deltas[i]

        # 4) Atualiza os pesos W1 e biases b1 (entre entrada e oculta)
        for i in range(self.n_hidden):
            for j in range(self.n_input):
                self.W1[i][j] += self.lr * hidden_deltas[i] * x[j]
            self.b1[i] += self.lr * hidden_deltas[i]

    # -------------------------------------------------------------------------
    # TREINAMENTO
    # -------------------------------------------------------------------------
    def train(self, X, Y, epochs):
        """
        Executa o treinamento por épocas ('epochs').
        Em cada época, percorre todo o conjunto de treino, fazendo
        forward + backward para cada amostra (Stochastic Gradient Descent)
        """
        for epoch in range(epochs):
            erro_total = 0.0
            for x, y in zip(X, Y):
                self.foward(x)
                # Erro Quadrático Médio (MSE) apenas para acompanhamento
                erro_total += sum((y[i] - self.output[i]) ** 2
                                    for i in range(self.n_output))
                self.backward(x, y)

                # Mostra o progresso a cada 100 épocas
                if (epoch + 1) % 100 == 0:
                    print(f"Época {epoch + 1:4d}/{epochs}  |  Erro total: {erro_total:.4f}")

    # -------------------------------------------------------------------------
    # PREDIÇÃO
    # -------------------------------------------------------------------------
    def predict(self, x):
        """
        Faz a propagação para frente e retorna o índice do neurônio
        de saída com maior ativação (a classe predita)
        """
        saida = self.foward(x)
        return  saida.index(max(saida))

# =============================================================================
# 4) FUNÇÃO PRINCIPAL: prepara os dados, treina e avalia
# =============================================================================
def main():
    # Fixa a semente para que o resultado seja reprodutível
    random.seed(42)

    # ---- Separa atributos (X) e rótulos (y) ----
    X = [linha[:-1] for linha in iris_data]     # 4 primeiros valores
    y = [int(linha[-1]) for linha in iris_data] # Último valor (classe)

    # ---- Pré-processamento ----
    X = normalize_features(X)       # Normalização min-max para [0, 1]
    Y = [one_hot_encode(c, 3) for c in y]       # Rótulos no formato one-hot

    # ---- Divisão treinamento/teste (80/20) com embaralhamento ----
    indices = list(range(len(X)))
    random.shuffle(indices)
    corte = int(0.8 * len(X))
    idx_treino, idx_teste = indices[:corte], indices[corte:]

    X_treino = [X[i] for i in idx_treino]
    Y_treino = [Y[i] for i in idx_treino]
    X_teste = [X[i] for i in idx_teste]
    y_teste = [y[i] for i in idx_teste]

    print("=" * 60)
    print(f"Total de amostras : {len(X)}")
    print(f"Treino            : {len(X_treino)}")
    print(f"Teste             : {len(X_teste)}")
    print("=" * 60)

    # ---- Cria e treina a rede ----
    # 4 entradas, 8 neurônios na camada oculta e 3 neurônios na saída (1 neurônio por classe)
    rede = MLP(n_input=4, n_hidden=8, n_output=3, learning_rate=0.1)

    print("\n>>> Iniciando o treinamento... \n")
    rede.train(X_treino, Y_treino, epochs = 100)

    # ---- Avaliação no conjunto de teste ----
    print("\n>>> Avaliando o conjunto de teste...\n")
    acertos = 0
    nomes_classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    for x, classe_real in zip(X_teste, y_teste):
        classe_pred = rede.predict(x)
        marca = "OK " if classe_pred == classe_real else " X "
        if classe_pred == classe_real:
            acertos += 1
        print(f"[{marca}] Esperado: {nomes_classes[classe_real]:<16}"
              f" | Predito: {nomes_classes[classe_pred]}")


    acuracia = acertos / len(y_teste) * 100
    print("\n" + "=" * 60)
    print(f"Acurácia no conjunto de teste: {acuracia:.2f}% "
          f"({acertos}/{len(y_teste)})")
    print("=" * 60)

# Ponto de entrada do programa
if __name__ == "__main__":
    main()