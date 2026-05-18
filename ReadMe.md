# 🔬 Detecção de Câncer por Análise de Imagens Médicas

> Projeto acadêmico desenvolvido com o objetivo de aplicar machine learning na classificação de imagens médicas para auxílio no diagnóstico de câncer.

---

## 📌 Sobre o Projeto

Este projeto consiste na criação de um modelo de machine learning capaz de analisar imagens médicas e classificá-las como **benignas ou malignas**, contribuindo para o processo de diagnóstico precoce de câncer.

O modelo é construído utilizando redes neurais convolucionais (CNN), passando por todas as etapas de um pipeline real de ciência de dados: coleta e preparação dos dados, análise exploratória, treinamento do modelo, avaliação de desempenho e interface de demonstração.

---

## ❓ Perguntas que o Projeto Busca Responder

- É possível, a partir de uma imagem médica, prever se uma lesão é benigna ou maligna?
- Quais características visuais de uma imagem estão mais associadas à presença de câncer?
- Com qual nível de confiança o modelo consegue fazer esse diagnóstico?
- O modelo comete mais erros ao classificar um caso maligno como benigno (falso negativo) ou o contrário?
- Como o desbalanceamento entre casos positivos e negativos impacta o desempenho do modelo?

---

## 🗂️ Dataset

**Fonte principal:** [ISIC Archive — Kaggle](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

| Atributo | Detalhe |
|---|---|
| Tipo de imagem | Dermoscópica (câncer de pele) |
| Classificação | Binária — Maligno / Benigno |
| Volume | ~25.000 imagens rotuladas |
| Formato | `.jpg` com metadados em `.csv` |
| Origem | ISIC Challenge (benchmark internacional) |

**Dataset alternativo:** [BreakHis — UCI / Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) *(histopatológico de mama, origem brasileira — UFPR)*

---

## 🔎 Variáveis Analisadas

### Variáveis da Imagem (extraídas pela CNN)
| Variável | Descrição |
|---|---|
| Padrão de textura | Distribuição e regularidade dos pixels na região da lesão |
| Bordas da lesão | Irregularidade e nitidez dos contornos |
| Coloração | Variação de tons e distribuição de cores na lesão |
| Simetria | Grau de simetria da forma da lesão |
| Tamanho relativo | Proporção da lesão em relação à imagem total |

### Variáveis de Entrada do Modelo
| Variável | Tipo | Descrição |
|---|---|---|
| `image_array` | Array numérico (224x224x3) | Imagem normalizada e redimensionada |
| `label` | Binário (0 / 1) | 0 = Benigno, 1 = Maligno |

### Variáveis de Saída (Predição)
| Variável | Descrição |
|---|---|
| `probabilidade` | Score entre 0 e 1 indicando a chance de malignidade |
| `classificação` | Benigno ou Maligno (baseado no threshold definido) |

---

## 🛠️ Stack Tecnológica

### Ambiente
| Ferramenta | Uso |
|---|---|
| Python 3.10+ | Linguagem principal |
| Google Colab | Ambiente de treino com GPU gratuita |
| Jupyter Notebook | Desenvolvimento e análise local |
| Git + GitHub | Versionamento do projeto |

### Bibliotecas
| Biblioteca | Finalidade |
|---|---|
| `pandas` / `numpy` | Manipulação de dados e arrays de imagem |
| `matplotlib` / `seaborn` | Visualização, EDA e gráficos de avaliação |
| `scikit-learn` | Métricas, divisão de dados e modelos baseline |
| `TensorFlow` / `Keras` | Construção e treinamento da CNN |
| `OpenCV` | Pré-processamento de imagens |
| `albumentations` | Data augmentation para imagens médicas |
| `Gradio` | Interface interativa para demonstração |

---

## 🔄 Pipeline do Projeto

```
1. Coleta e carregamento do dataset (Kaggle)
        ↓
2. Análise exploratória (EDA) — distribuição de classes, exemplos visuais
        ↓
3. Pré-processamento — resize 224x224, normalização, remoção de artefatos
        ↓
4. Data Augmentation — rotação, flip, zoom, variação de brilho
        ↓
5. Divisão estratificada — Train / Validation / Test (70/15/15)
        ↓
6. Construção da CNN do zero (Conv2D → BatchNorm → Pooling → Dropout → Dense)
        ↓
7. Treinamento com Early Stopping e ajuste de pesos por classe
        ↓
8. Avaliação — AUC-ROC, F1-Score, Sensitivity, Specificity, Confusion Matrix
        ↓
9. Visualização Grad-CAM — regiões da imagem usadas pelo modelo
        ↓
10. Interface Gradio — demonstração com upload de imagem
```

---

## 📊 Métricas de Avaliação

> ⚠️ Em contexto médico, **falso negativo é mais grave que falso positivo**. Um caso maligno classificado como benigno pode atrasar um diagnóstico crítico. As métricas foram escolhidas com esse risco em mente.

| Métrica | Por que usamos |
|---|---|
| **AUC-ROC** | Avalia o desempenho geral independente do threshold |
| **Sensitivity (Recall)** | Mede quantos casos malignos o modelo detecta corretamente |
| **Specificity** | Mede quantos casos benignos são classificados corretamente |
| **F1-Score** | Equilíbrio entre precisão e recall em dados desbalanceados |
| **Confusion Matrix** | Visão completa dos acertos e erros por classe |

> A métrica **accuracy** não é usada como indicador principal por ser enganosa em datasets desbalanceados.

---

## 📁 Estrutura do Repositório

```
├── data/
│   ├── raw/              # Imagens originais do dataset
│   └── processed/        # Imagens pré-processadas
├── notebooks/
│   ├── 01_eda.ipynb             # Análise exploratória
│   ├── 02_preprocessing.ipynb   # Pré-processamento e augmentation
│   ├── 03_model.ipynb           # Construção e treinamento da CNN
│   └── 04_evaluation.ipynb      # Avaliação e Grad-CAM
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
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute os notebooks na ordem (pasta /notebooks)

# 4. Para rodar a interface de demonstração
python app/app.py
```

---

## 👥 Equipe

| Nome | GitHub |
|---|---|
| Lucas Hahne  | @Dev-LucasHahne |
| Lucas Marques | @Lmsantozio |
| Nome  | User do GitHub |
| Nome  | User do GitHub |



---

## 🏫 Informações Acadêmicas

- **Instituição:** Universidade São Judas Tadeu
- **Curso:** Ciência da Computação
- **Disciplina:** *Inteligência Artificial*
- **Professor:** *Prof. Jessé Guimarães*