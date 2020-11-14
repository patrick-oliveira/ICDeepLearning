# ICDeepLearning

Repositório contendo os arquivos para o processamento, análise e classificação de sinais de EEG no contexto de Interfaces Cérebro-Computador (SSVEP) utilizando redes convolucionais e técnicas de imageamento de séries temporais como uma etapa de pré-processamento. A classificação utilizou os modelos pré-treinados disponibilizados no pacote PyTorch, a fim de tirar proveito de Transfer Learning. O projeto também propõe a construção de uma arquitetura de autoencoder baseado em módulos convolucionais para a codificação dos sinais em imagens (tensores de três canais). 

Os scripts e notebooks estão organizados de tal modo que podem funcionar sem a necessidade de ajustes desde que se tenha acesso ao repositório dos dados. Requisições da base de dados utilizada neste projeto podem ser feitas para o email denis.fantinato@ufabc.edu.br.

O projeto foi conduzido no contexto de um edital de iniciação científica na UFABC, pelo período de agosto de 2019 e junho de 2020, com o financiamento da CNPq. Uma descrição mais aprofundada do projeto, fundamentos de BCI e aquisição dos dados, considerando ainda os métodos de imageamento, a arquitetura de autoencoder e os classificadores, juntamente com os resultados obtidos neste período, pode ser encontrado no arquivo "Relatório_CNPQ".
