# Aqui apresento uma rede neural de perceptron de uma camada
# quero com esse trabalho mostrar o quão é fundamental saber o que acontece por detrás das ferramentas como: scikit learn, Tensorflow etc... 
import numpy as np


# variaveis de entradas
input = np.array([[0,0], [0,1], [1,0], [1,1]])
# saídas alvos, pois se trata de um aprendizado supervisionado
saídas_esperadas = np.array([0,0,0,1]) 
# Pesos iniciais do modelo
pesos = np.array([0.0, 0.0]) 
aprendizagem = 0.1

# A função de ativação usada é a linear, funciona muito boa para situações linearmente separaveis
def funcao_ativacao(soma):
    if(soma >=1):
        return 1
    else:
        return 0

# Saída calculada pelo modelo
def cal_Saida(registro):
    s = registro.dot(pesos)
    return funcao_ativacao(s)   

# Processo de treinamento
def fit():
    erro_medio = 1
    # O treino termina quando o erro_medio for igual zero
    # Salientar ainda determinei o erro_medio igual zero por ser uma base de dados muito pequena, porque em bigdata é praticamente impossivel 
    while (erro_medio != 0):
         erro_medio = 0
         for i in range(len(saídas_esperadas)):
             saída_modelo = cal_Saida(np.array(input[i]))
             erro = abs(saídas_esperadas[i] - saída_modelo)
             erro_medio += erro
             for j in range(len(pesos)):
                 # Atualização de pesos até que o erro seja zero para melhorar performance 
                 pesos[j] = pesos[j] + (aprendizagem*input[i][j]*erro)
                 print(f"Peso atualizado: ", pesos[j])
         print(f"erro_total: ", erro_medio)

fit()         

