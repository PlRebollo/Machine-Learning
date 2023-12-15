library(readr)
library(tidyverse)
library(ranger) # pacote para o ramdon forest, não utilizarei o pacote RamdomForest pois este é mais rápido
library(gbm)# boosting

spam <-read.csv("spambase_csv.csv")

indices_treino <- sample(1:nrow(spam), 0.7 * nrow(spam))
spam_treino <- spam[indices_treino, ]
spam_teste <- spam[-indices_treino, ]

# Defina uma semente para reproduzibilidade
set.seed(217460)

######################### Random Forest

floresta <- ranger(as.factor(class) ~. ,data = spam_treino, importance = "impurity")

predito_floresta <- predict(floresta, data = spam_teste)

# calculando a importancia das variáveis
importance <- tibble(variavel = names(importance(floresta)), importancia = importance(floresta)) %>% 
  arrange(desc(importancia))

ggplot(importance %>% top_n(20), aes(x = reorder(variavel, importancia), y = importancia)) + 
  geom_bar(stat = "identity", position = "dodge")+
  coord_flip()

# matriz de confusão
matriz_confusao_floresta <- table(Real = spam_teste$class, Predito = predito_floresta$predictions)
matriz_confusao_floresta

######################### Bagging

bagging <- ranger(as.factor(class) ~. ,data = spam_treino, importance = "impurity", mtry = 57)

predito_bagging <- predict(bagging, data = spam_teste)

# calculando a importancia das variáveis
importance_bagging <- tibble(variavel = names(importance(bagging)), importancia = importance(bagging)) %>% 
  arrange(desc(importancia))

ggplot(importance_bagging %>% top_n(20), aes(x = reorder(variavel, importancia), y = importancia)) + 
  geom_bar(stat = "identity", position = "dodge")+
  coord_flip()

#matriz de confusão
matriz_confusao_bagging <- table(Real = spam_teste$class, Predito = predito_bagging$predictions)
matriz_confusao_bagging


######################### Boosting

boosting <- gbm::gbm(class ~. ,data = spam_treino, distribution = "bernoulli",
                n.trees = 500, interaction.depth = 4)


predito_boosting <- predict(boosting, newdata = spam_teste, type = "response")

predito_boosting <- ifelse(predito_boosting >= 0.5, 1, 0)


#matriz de confusão
matriz_confusao_boosting <- table(Real = spam_teste$class, Predito = predito_boosting)
matriz_confusao_boosting


