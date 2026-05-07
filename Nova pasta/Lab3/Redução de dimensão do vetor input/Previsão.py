# Atividade 4: Código de Predição
import numpy as np

from Treino import X_test, knn, y_test, pca_train


def predizer_digito(novo_dado, model, pca_model):
    # O novo dado deve ter 64 características (8x8 pixels)
    # Primeiro reduzimos a dimensão com o PCA já treinado
    dado_reduzido = pca_model.transform(novo_dado.reshape(1, -1))

    # Realizamos a predição
    predicao = model.predict(dado_reduzido)
    probabilidade = model.predict_proba(dado_reduzido)

    return predicao[0], np.max(probabilidade)


# Exemplo de teste com uma amostra do dataset
exemplo_indice = 10
resultado, confianca = predizer_digito(X_test[exemplo_indice], knn, pca_train)

print(f"Dígito Real: {y_test[exemplo_indice]}")
print(f"Dígito Predito: {resultado} (Confiança: {confianca:.2%})")