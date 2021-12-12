# Trilha de Estudos de Ciência de Dados

Este é um documento em constante construção e atualização. Adiciono aqui técnicas de estudo e fontes que considero boas para o aprendizado de ciência de dados, com o objetivo de manter recursos organizados para consulta e ajudar quem se interessa pelo tema. O conteúdo vai do básico ao avançado.

# Sumário
- [Recomendações sobre como estudar](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#recomendações-sobre-como-estudar)
- [Ferramentas](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#ferramentas)
- [Python and Data Analysis basics](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#python-and-data-analysis-basics)
- [Data Visualization](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#data-visualization)
- [Machine Learning - Teoria](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#machine-learning---teoria)
- [Machine Learning - Prática](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#machine-learning---prática)
- [Deep Learning - Neural Networks](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#deep-learning---neural-networks)
- [Transformers](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#transformers)
- [NLP - Natural Language Processing](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#nlp---natural-language-processing)
- [Computer Vision](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#computer-vision)
- [Youtube channels](https://github.com/HinePo/Trilha-de-Estudos-de-Data-Science/blob/main/README.md#youtube-channels)

# Recomendações sobre como estudar
- Criar um doc (word) pessoal com a sua organização do que vc já aprendeu/estudou e o que planeja aprender/estudar, de preferência organizado por mês ou bimestre. Procurar manter este doc atualizado, se possível;
- Instalar [video speed controller](https://chrome.google.com/webstore/detail/video-speed-controller/nffaoalbilbmmfgbnbgppjihopabppdk) no google chrome (funciona em qualquer vídeo tocado pelo chrome browser), e aprender a usar:
![image](https://user-images.githubusercontent.com/66163270/145697555-17f7fb51-ec8d-4f9f-8c36-654b062cddce.png)
-	Ao entrar em um assunto novo, gosto de ver um ou dois vídeos de ~10 min no youtube, pesquisar sobre o tema focando em material escrito, e estudar aplicações;
-	Evitar ficar muito tempo na parte teórica: Qualquer assunto novo tem suas aplicações, via bibliotecas específicas. Se familiarizar com a documentação é o primeiro passo (google “pandas docs”, “seaborn cheat sheet”...);
-	O segundo passo é a aplicação e uso, parte prática: Resolver problemas usando IA: Pesquisar aplicações no Kaggle (notebooks), fazer o fork, adicionar ideias. Evitar tentar reinventar a roda: aproveitar os códigos que já existem;
- Adicionar aplicação ao seu repositório pessoal (público ou privado).

# Ferramentas
- Focar em Google Colab e Kaggle notebooks.
- No futuro, é interessante conhecer IDEs como VS Code, PyCharm e Spyder.
- Sublime Text é um ótimo editor de código.

# Python and Data Analysis basics
- [Never memorize code](https://www.youtube.com/watch?app=desktop&v=AavXBoxTCIA) - vídeo
- [How to learn data science smartly](https://www.youtube.com/watch?app=desktop&v=csG_qfOTvxw) - vídeo
- [Didática Tech playlists](https://www.youtube.com/c/Did%C3%A1ticaTech/playlists?app=desktop)
- [Curso em vídeo playlists](https://www.youtube.com/c/CursoemV%C3%ADdeo/playlists?app=desktop)
- [Python projects](https://medium.com/coders-camp/180-data-science-and-machine-learning-projects-with-python-6191bc7b9db9)
- [Learn Pandas with pokemons](https://www.kaggle.com/ash316/learn-pandas-with-pokemons)
- [Pandas docs](https://pandas.pydata.org/docs/index.html)

Data Analysis workflow - entender e praticar as etapas básicas:

- Importar e ler csv, criar dataframe
- Checar tipos de variáveis (data types): numéricas e categóricas
- Preproces: Técnicas para lidar com variáveis categóricas: one-hot encoding, label encoding, ordinal encoding....
- Plots básicos
- Analisar missing values (valores faltantes), tomar decisões sobre o que fazer com eles
- Analisar outliers, decidir o que fazer com eles
- Análise univariada, bivariada, multivariada
- Feature Engineering (criação de variáveis)
- Deixar dados prontos para eventual modelagem de IA

Machine Learning workflow - entender e praticar as etapas básicas:

- Split train/test datasets
- Definir Features and Target (if it is a supervised problem)
- Preprocess: Scaling
- Definir métricas de avaliação dos modelos
- Choose algorithm, Train model
- Evaluate model
- Melhorar modelo, tunar hiperparâmetros, treinar de novo, avaliar de novo
- Ensemble: combinar modelos para aumentar performance e poder de generalização

# Data Visualization
- [A Simple Tutorial To Data Visualization](https://www.kaggle.com/vanshjatana/a-simple-tutorial-to-data-visualization#notebook-container) - @vanshjatana
- [Séries de notebooks de visualização](https://www.kaggle.com/residentmario/univariate-plotting-with-pandas) - ao final de cada notebook tem um link para o próximo
- [Data Visualization & Prediction](https://www.kaggle.com/hinepo/pnad-data-analysis) - @hinepo
- [Power BI playlists](https://www.youtube.com/c/HashtagTreinamentos/playlists?app=desktop)
- [Power BI - Karine Lago](https://www.youtube.com/c/KarineLago/playlists?app=desktop)
- [Power BI + DAX + Projetos na prática - Curso Udemy](https://www.udemy.com/course/curso-de-powerbi-desktop-dax/)

# Machine Learning - Teoria
- [Supervised x Unsupervised Learning](https://www.youtube.com/watch?app=desktop&v=1FZ0A1QCMWc) - vídeo
- [Supervised x Unsupervised Learning: applications](https://www.youtube.com/watch?app=desktop&v=rHeaoaiBM6Y) - vídeo
- Pesquisar sobre Overfitting e Underfitting, ver vídeos e gráficos
- [Cross Validation](https://www.youtube.com/watch?app=desktop&v=fSytzGwwBVw)
- [Cross Validation - scikit docs](https://scikit-learn.org/stable/modules/cross_validation.html)
- Pesquisar sobre Cross Validation para Time Series (como evitar contaminação de dados do futuro pro passado, data leakage, train/test contamination...)
- [pdf do livro do Abhishek Thakur](https://github.com/abhishekkrthakur/approachingalmost/blob/master/AAAMLP.pdf) - resumo com tudo, disponível na Amazon tb
- [Kaggle courses](https://www.kaggle.com/learn)
- [Statquest - Vídeos sobre conceitos, teoria e matemática de algoritmos e ML](https://www.youtube.com/c/joshstarmer/playlists?app=desktop)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - Acho muito importante ler todo o item 1. Na primeira leitura não precisa entender tudo com profundidade, mas tem que se familiarizar com a documentação do scikit, especialmente com o item 1 todo. É uma biblioteca muito importante para se aprender a usar e consultar.
- [Scikit-learn Pre-processing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [ML projects for beginners - with code](https://github.com/microsoft/ML-For-Beginners)
- [Scipy docs](https://scipy.org/) - Procurar aplicação do pacote
- Pesquisar sobre "Feature Engineering" (criação de variáveis)
- Pesquisar sobre métricas e como avaliar modelos:
  - Classificação: Accuracy, ROC AUC, f1-score, recall, precision
  - Regressão: RMSE, MSE, MAE, R²
- Outros conceitos importantes: Pesquisar sobre Boosting (XGBoost, LGBM, GBM), Bagging, Split train/test, data leakage, time series, ARIMA, feature importances, ensemble...
- Imbalanced learning:
  - downsampling/upsampling
  - [Transforming skewed data](https://medium.com/@ODSC/transforming-skewed-data-for-machine-learning-90e6cc364b0) - como tratar o viés no dados
  - [imblearn](https://opendatascience.com/strategies-for-addressing-class-imbalance/)
  - [Oversampling x Undersampling](https://www.kdnuggets.com/2020/01/5-most-useful-techniques-handle-imbalanced-datasets.html)
  - [Resampling example](https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
  - [SMOTE for classification example](https://www.kaggle.com/shrutimechlearn/pokemon-classification-and-smote-for-imbalance/notebook)

# Machine Learning - Prática
- [Kaggle's 30 Days of ML](https://www.youtube.com/playlist?app=desktop&list=PL98nY_tJQXZnP-k3qCDd1hljVSciDV9_N) - Abhishek Thakur
- [Applied Machine Learning](https://www.kaggle.com/vanshjatana/applied-machine-learning) - @vanshjatana
- Browse kaggle, ver notebooks e datasets dos assuntos que te interessam
- Fazer forks de notebooks do kaggle (Copy and Edit), testar hipóteses e técnicas
- Falar com as pessoas do kaggle, comentar e postar, fazer parte da comunidade
- Competições 'Getting Started': estudar notebooks com bom score, e usar técnicas e conceitos aprendidos para criar o seu próprio. Estudar notebooks com score médio, comparar com os de score bom, e entender o que causou a melhora na pontuação. Recomendo no mínimo uns 10 dias de estudo para cada uma das competições abaixo:
  - [Titanic Classification](https://www.kaggle.com/c/titanic)
  - [House Prices Regression](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  - [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
  - [Tabular Playground Series](https://www.kaggle.com/c/tabular-playground-series-sep-2021)
  - Nível avançado: competições reais (valendo prêmios)

# Deep Learning - Neural Networks
Principais conceitos e keywords a pesquisar e aprender: tensors, gradient descent, automatic differentiation, forward pass, backpropagation, layers, vanishing gradients, exploding gradients, fine-tuning, transfer learning...

- [Aula Intro de DL - Lex Friedman](https://www.youtube.com/watch?app=desktop&v=O5xeyoRL95U) - vídeo
- [Keras docs](https://keras.io/)
- [Keras Sequential class](https://keras.io/api/models/sequential/)
- [Tensorflow docs](https://www.tensorflow.org/)
- [Pytorch docs - getting started](https://pytorch.org/get-started/locally/)
- [Pytorch - abhishek Thakur playlist and tutorials](https://www.youtube.com/playlist?app=desktop&list=PL98nY_tJQXZln8spB5uTZdKN08mYGkOf2)
- [Pytorch - torch.nn](https://pytorch.org/docs/stable/nn.html)

Um estudo muito útil e proveitoso é comparar e olhar em paralelo as documentações de Quick Start do Keras, do Tensorflow e do Pytorch. A lógica é bem parecida e existem muitas analogias:
- [Keras Quick Start](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Tensorflow Quick Start](https://www.tensorflow.org/tutorials/quickstart/advanced)
- [Pytorch Quick Start](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

Principais tipos de camadas (layers):
- Dense & Linear (fully connected)
- Activation functions (ReLU, LeakyReLU, SELU, PReLU, Tanh, Softmax....)
- Conv (Convolutional)
- Flatten
- Batch Normalization
- LSTM (Long Short Term Memory)
- GRU (Gated Recurrent Unit - Short Term Memory)
- Dropout
- Pooling

# Transformers


# NLP - Natural Language Processing


# Computer Vision



# Youtube channels



