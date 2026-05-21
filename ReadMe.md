# 🎗️ Detecção de Câncer de Mama por Análise de Imagens Histopatológicas

> Projeto acadêmico desenvolvido na Universidade São Judas Tadeu com o objetivo de aplicar machine learning na classificação de imagens histopatológicas de tecido mamário para auxílio no diagnóstico de câncer de mama.

---

## 📌 Sobre o Projeto

O câncer de mama é o tipo de câncer mais comum entre mulheres no Brasil e no mundo. O diagnóstico precoce é o principal fator que aumenta as chances de tratamento bem-sucedido — e é exatamente nesse ponto que este projeto atua.

A partir de imagens de biópsias de tecido mamário, treinamos um modelo de deep learning capaz de identificar automaticamente se o tecido analisado é **benigno ou maligno**. O modelo é construído do zero com redes neurais convolucionais (CNN), cobrindo todas as etapas de um pipeline real: preparação dos dados, treinamento, avaliação e demonstração interativa.

---

## ❓ Perguntas que o Projeto Busca Responder

- É possível identificar câncer de mama a partir de imagens histopatológicas com um modelo de deep learning?
- Quais padrões visuais do tecido mamário estão associados à malignidade?
- Com que nível de confiança o modelo consegue fazer essa classificação?
- Como o modelo se comporta diante de casos limítrofes — tecidos difíceis de classificar?
- O modelo comete mais erros em falsos negativos (maligno classificado como benigno) ou falsos positivos?
- O aumento de imagens (data augmentation) melhora significativamente o desempenho em datasets pequenos?

---

## 🗂️ Dataset

**Dataset principal:** [BreakHis — Breast Cancer Histopathological Database](https://www.kaggle.com/datasets/ambarish/breakhis)

> Dataset de origem **brasileira**, desenvolvido pelo Laboratório de Visão Robótica e Imagem (VRI) da **UFPR — Universidade Federal do Paraná**.

| Atributo               | Detalhe                                     |
| ---------------------- | ------------------------------------------- |
| Tipo de imagem         | Histopatológica (biópsia de tecido mamário) |
| Classificação          | Binária — Maligno / Benigno                 |
| Volume                 | ~7.909 imagens rotuladas                    |
| Ampliações disponíveis | 40x, 100x, 200x, 400x                       |
| Formato                | `.png` colorido (RGB)                       |
| Fonte                  | UFPR / Kaggle / UCI ML Repository           |

### Distribuição das Classes

| Classe      | Subtipos incluídos                                                           | Qtd. aproximada |
| ----------- | ---------------------------------------------------------------------------- | --------------- |
| **Benigno** | Adenose, Fibroadenoma, Tumor Phyllodes, Papiloma Tubular                     | ~2.480 imagens  |
| **Maligno** | Carcinoma Ductal, Carcinoma Lobular, Carcinoma Mucinoso, Carcinoma Papilário | ~5.429 imagens  |

> ⚠️ O dataset é **desbalanceado** (mais casos malignos). Isso será tratado com pesos de classe e métricas adequadas.

---

## 🔎 Variáveis Analisadas

### Características Visuais do Tecido (aprendidas pela CNN)

| Variável                    | Descrição                                                                       |
| --------------------------- | ------------------------------------------------------------------------------- |
| Organização celular         | Distribuição e regularidade das células no tecido                               |
| Tamanho e forma dos núcleos | Núcleos irregulares e aumentados são indicativos de malignidade                 |
| Densidade do tecido         | Quantidade de células por área na imagem                                        |
| Coloração (H&E staining)    | Padrão de coloração hematoxilina-eosina característico de cada tipo de tecido   |
| Presença de mitoses         | Células em divisão são um marcador importante de tumor maligno                  |
| Arquitetura glandular       | Estrutura das glândulas mamárias — preservada no benigno, distorcida no maligno |

### Variáveis de Entrada do Modelo

| Variável        | Tipo                                  | Descrição                           |
| --------------- | ------------------------------------- | ----------------------------------- |
| `image_array`   | Array numérico (224x224x3)            | Imagem normalizada e redimensionada |
| `magnification` | Categórico (40x / 100x / 200x / 400x) | Nível de ampliação da imagem        |
| `label`         | Binário (0 / 1)                       | 0 = Benigno, 1 = Maligno            |

### Variáveis de Saída (Predição)

| Variável        | Descrição                                           |
| --------------- | --------------------------------------------------- |
| `probabilidade` | Score entre 0 e 1 indicando a chance de malignidade |
| `classificação` | Benigno ou Maligno (definido pelo threshold)        |

---

## 🛠️ Stack Tecnológica

### Ambiente

| Ferramenta       | Uso                             |
| ---------------- | ------------------------------- |
| Python 3.10+     | Linguagem principal             |
| Google Colab     | Treinamento com GPU T4 gratuita |
| Jupyter Notebook | Desenvolvimento e análise local |
| Git + GitHub     | Versionamento do projeto        |

### Bibliotecas

| Biblioteca               | Finalidade                                           |
| ------------------------ | ---------------------------------------------------- |
| `pandas` / `numpy`       | Manipulação de metadados e arrays de imagem          |
| `matplotlib` / `seaborn` | EDA, curvas de treino e gráficos de avaliação        |
| `scikit-learn`           | Métricas, divisão estratificada dos dados e baseline |
| `TensorFlow` / `Keras`   | Construção e treinamento da CNN do zero              |
| `OpenCV`                 | Pré-processamento e análise das imagens              |
| `albumentations`         | Data augmentation para imagens histopatológicas      |
| `Gradio`                 | Interface interativa para demonstração do modelo     |

---

## 🔄 Pipeline do Projeto

```
1. Coleta do dataset BreakHis (Kaggle / UFPR)
        ↓
2. Análise exploratória (EDA)
   — distribuição benigno/maligno, variação por ampliação, exemplos visuais
        ↓
3. Pré-processamento
   — resize 224x224, normalização [0,1], separação por ampliação
        ↓
4. Data Augmentation
   — rotação, flip horizontal/vertical, zoom, variação de brilho e contraste
        ↓
5. Divisão estratificada — Train / Validation / Test (70 / 15 / 15)
        ↓
6. Construção da CNN do zero
   — Conv2D → BatchNormalization → MaxPooling → Dropout → Dense → Sigmoid
        ↓
7. Treinamento
   — class_weight para desbalanceamento, Early Stopping, ReduceLROnPlateau
        ↓
8. Avaliação
   — AUC-ROC, F1-Score, Sensitivity, Specificity, Confusion Matrix
        ↓
9. Explicabilidade — Grad-CAM
   — mapa de calor sobre a imagem mostrando o que o modelo "vê"
        ↓
10. Interface Gradio
    — upload de imagem → classificação ao vivo com probabilidade e Grad-CAM
```

---

## 📊 Métricas de Avaliação

> ⚠️ Em diagnóstico de câncer, **falso negativo é o pior erro possível** — classificar um tumor maligno como benigno pode atrasar um tratamento crítico. As métricas foram escolhidas com esse risco em mente.

| Métrica                  | Por que usamos                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------- |
| **Sensitivity (Recall)** | Mede quantos tumores malignos o modelo detecta corretamente — **métrica prioritária** |
| **Specificity**          | Mede quantos tecidos benignos são corretamente identificados                          |
| **AUC-ROC**              | Avalia o desempenho geral do modelo independente do threshold                         |
| **F1-Score**             | Equilíbrio entre precisão e recall — importante com dados desbalanceados              |
| **Confusion Matrix**     | Visão completa dos acertos, falsos positivos e falsos negativos                       |

> A métrica **accuracy** não é usada como indicador principal — com o dataset desbalanceado do BreakHis, um modelo que classifica tudo como maligno já teria ~68% de accuracy.

---

## 📁 Estrutura do Repositório

```
├── data/
│   ├── raw/              # Imagens originais do BreakHis
│   └── processed/        # Imagens pré-processadas (224x224, normalizadas)
├── notebooks/
│   ├── 01_eda.ipynb              # Análise exploratória do dataset
│   ├── 02_preprocessing.ipynb    # Pré-processamento e augmentation
│   ├── 03_model.ipynb            # Construção e treinamento da CNN
│   └── 04_evaluation.ipynb       # Avaliação, Grad-CAM e resultados
├── src/
│   ├── model.py          # Arquitetura da CNN
│   ├── dataset.py        # Carregamento e divisão dos dados
│   ├── train.py          # Pipeline de treinamento
│   └── utils.py          # Funções auxiliares
├── app/
│   └── app.py            # Interface Gradio
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/cancer-mama-classificacao.git
cd cancer-mama-classificacao

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Faça o download do dataset BreakHis no Kaggle e coloque em data/raw/

# 4. Execute os notebooks na ordem (pasta /notebooks)

# 5. Para rodar a interface de demonstração
python app/app.py
```

---

## 👥 Equipe

| Nome          | GitHub          |
| ------------- | --------------- |
| Lucas Hahne   | @Dev-LucasHahne |
| Lucas Marques | @LmSantoz       |
| Danilo Soares | @Snoker68       |
| Flavio Tanaka | @Fww-t          |

---

## 🏫 Informações Acadêmicas

- **Instituição:** Universidade São Judas Tadeu
- **Curso:** Ciência da Computação
- **Disciplina:** _Inteligencia Artificial_
- **Professor:** Jessé Ferreira Guimaraes Netto\*\*
