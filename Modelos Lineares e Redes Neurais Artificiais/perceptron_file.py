import random
import matplotlib.pyplot as plt


# Definindo a função de ativação (degrau / Heaviside):
def step_function(x):
    return 1 if x >= 0 else 0

# Classe Perceptron simples com histórico de pesos:
class Perceptron:
    def __init__(self, input_size, learning_rate = 0.2):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.history = []       # Irá armazenar os pesos e bias de cada época.

    def predict(self, inputs):
        u = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return step_function(u)

    def train(self, training_data, epochs = 100):
        for epoch in range(epochs):
            print(f'Epoca: {epoch+1}')
            global_error = 0        # Soma dos erros da época.

            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction         # valor desejado - previsto (d - y)

                # Atualização dos pesos:
                self.weights = [
                    w + self.learning_rate * error * x              # w^(t+1) = w^t + nex_i
                    for w, x in zip(self.weights, inputs)
                ]
                self.bias += self.learning_rate * error          # b^(t+1) = b^(t) + nx_i

                global_error += abs(error)
                print(f"Entrada: {inputs}, Esperado: {target}, Previsto: {prediction}")

            # Salvando os pesos e bias (viés) atuais:
            self.history.append((self.weights[:], self.bias))

            # Exibição dos pesos e bias com três casas decimais:
            w_str = [f'{w:.3f}' for w in self.weights]
            b_str = f'{self.bias:.3f}'
            print(f'Pesos: {w_str}, Bias: {b_str}')
            print('-' * 30)
            # print(f"Pesos: {self.weights}, Bias: {self.bias}")
            # print('-' * 30)


            # Condição de parada após convergência:
            if global_error == 0:
                self.converged_epoch = epoch + 1
                print(f'O treinamento convergiu na epoca {epoch+1}')
                print(f'O treinamento foi concluido em {self.converged_epoch} epocas')
                break


    def get_decision_boundary(self):
        w1, w2 = self.weights
        b = self.bias

        if w2 == 0:
            return None
        # A equação da reta é w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
        return lambda x: -(w1 * x + b) / w2

# Criação dos dados de entrada ded acordo com as tabelas verdades das portas
# lógicas AND, OR e NAND.
def get_training_data(gate_type):
    if gate_type == 'AND':
        data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    elif gate_type == 'OR':
        data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    elif gate_type == 'NAND':
        data = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    else:
        raise ValueError('Porta lógica não definida ou suportada hehe.')

    return data, gate_type

# Informe a porta lógica aqui:
training_data, gate_name = get_training_data('AND')

# Criando e treinando a Perceptron com duas entradas:
p = Perceptron(input_size = 2)
p.train(training_data, epochs = 10)

# Visualiazção da evolução dos pesos:
w1_vals = [w[0][0] for w in p.history]
w2_vals = [w[0][1] for w in p.history]
bias_vals = [w[1] for w in p.history]

# Plotando o gráfico com matplotlib:
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(w1_vals, label='Peso w1')
plt.plot(w2_vals, label='Peso w2')
plt.plot(bias_vals, label='Bias')
plt.xlabel('Epoca')
plt.ylabel('Valor')
plt.title('Evolução dos Pesos e Bias durante o treinamento')
plt.legend()
plt.grid(True)

# Plot da reta de separação e pontos:
plt.subplot(1, 2, 2)
for x, y in training_data:
    color = 'blue' if y == 1 else 'red'
    plt.scatter(x[0], x[1], c=color, label=f'Classe {y}')


# Reta de decisão final
decision = p.get_decision_boundary()
if decision:
    x_vals = [i * 0.1 for i in range(-1, 12)]
    y_vals = [decision(x) for x in x_vals]
    plt.plot(x_vals, y_vals, 'k--', label='Fronteira de decisão')

plt.title(f'Porta lógica {gate_name} - Perceptron')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
