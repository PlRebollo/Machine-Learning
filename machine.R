library(glmnet)
library(readr)
library(MASS)
library(caret)
wine <- read_csv("Unicamp/machine/wine/wine.data", col_names = FALSE)

########
glm.fits=glm(X1 ~.,
             data=wine )
summary (glm.fits)

glm.fits=glm(X1 ~ X2  + X8 + X11 + X14,
             data=wine )

# modelo adotado, variáveis 2, 8, 11, 14
wine <- wine[,c(1,2, 8,11,14)]


# Defina uma semente para reproduzibilidade
set.seed(217460)


# Divida o conjunto de dados em treinamento (70%) e teste (30%)
indices_treino <- sample(1:nrow(wine), 0.5 * nrow(wine))
dados_treino <- wine[indices_treino, ]
dados_teste <- wine[-indices_treino, ]

# Divida os conjuntos de treinamento e teste em características (X) e classe (y)
X_treino <- model.matrix(factor(X1)~.,dados_treino)[,-1]  # Todas as colunas, exceto a primeira (classe)
y_treino <- dados_treino$X1  # Primeira coluna (classe)
X_teste <- model.matrix(factor(X1)~.,dados_teste)[,-1]    # Todas as colunas, exceto a primeira (classe)
y_teste <- dados_teste$X1     # Primeira coluna (classe)

# Crie um modelo de regressão logística penalizada com penalização Ridge (L2)
model_ridge <- glmnet(X_treino, as.factor(y_treino), family = "multinomial", alpha = 0)
#cross-validation para descobrir o melhor valor de s
cv_outL2=cv.glmnet(X_treino,as.factor(y_treino),alpha=0, family = "multinomial")
plot(cv_outL2)
bestlam =cv_outL2$lambda.min


# Crie um modelo de regressão logística penalizada com penalização Lasso (L1)
model_lasso <- glmnet(X_treino, as.factor(y_treino), family = "multinomial", alpha = 1)
#cross-validation para descobrir o melhor valor de s
cv_outL1=cv.glmnet(X_treino,as.factor(y_treino),alpha=1, family = "multinomial")
plot(cv_outL1)
bestlam_L1 =cv_outL1$lambda.min




# Faça previsões nos dados de teste
previsoes_ridge <- predict(model_ridge, newx = X_teste, type = "response", s = bestlam)
previsoes_lasso <- predict(model_lasso, newx = X_teste, type = "response", s = bestlam_L1)

# Converta as previsões em classes
classe_pred_ridge <- colnames(previsoes_ridge)[apply(previsoes_ridge, 1, which.max)]
classe_pred_lasso <- colnames(previsoes_lasso)[apply(previsoes_lasso, 1, which.max)]

# Calcular a matriz de confusão para Ridge
matriz_confusao_ridge <- table(Real = y_teste, Predito = classe_pred_ridge)
print("Matriz de Confusão para Ridge:")
print(matriz_confusao_ridge)

# Calcular a matriz de confusão para Lasso
matriz_confusao_lasso <- table(Real = y_teste, Predito = classe_pred_lasso)
print("Matriz de Confusão para Lasso:")
print(matriz_confusao_lasso)

##############################
#LDA e LQA

# Ajuste o modelo LDA
modelo_lda <- lda(as.factor(X1) ~ ., data = wine, subset = indices_treino)

# Ajuste o modelo QDA
modelo_qda <- qda(as.factor(X1) ~ ., data = dados_treino)

# Faça previsões usando LDA
previsoes_lda <- predict(modelo_lda, newdata = dados_teste)

# Faça previsões usando QDA
previsoes_qda <- predict(modelo_qda, newdata = dados_teste)

# Avalie o desempenho dos modelos (por exemplo, acurácia)
acuracia_lda <- sum(previsoes_lda$class == y_teste) / length(y_teste)
acuracia_qda <- sum(previsoes_qda$class == y_teste) / length(y_teste)

print("Acurácia do modelo LDA:")
print(acuracia_lda)

print("Acurácia do modelo QDA:")
print(acuracia_qda)


matriz_confusao_lda <- confusionMatrix(previsoes_lda$class, as.factor(y_teste))
print("Matriz de Confusão para LDA:")
print(matriz_confusao_lda)

matriz_confusao_qda <- confusionMatrix(previsoes_qda$class, as.factor(y_teste))
print("Matriz de Confusão para QDA:")
print(matriz_confusao_qda)



