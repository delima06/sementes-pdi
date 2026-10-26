# Projeto de PDI - Identificação, Segmentação e Contagem de Sementes

Projeto desenvolvido para a disciplina **TAD0018 - Processamento Digital de Imagens (Unidade II)**  
**Curso:** Tecnologia em Análise e Desenvolvimento de Sistemas — EAJ / UFRN.

---

## 🎯 Objetivo
Desenvolver e avaliar um método computacional em Python com OpenCV para identificação, segmentação e contagem automática de grãos de feijão sobre superfície de EVA verde, além de extrair métricas físicas individuais (área, perímetro, circularidade e bounding boxes).

---

## 📁 Estrutura do Repositório

```text
pdi-sementes/
├── dados/
│   ├── imagens/
│   │   ├── desenvolvimento/    # 10 imagens para calibrar e testar o algoritmo (dev_01 a dev_10)
│   │   └── generalizacao/      # 10 imagens reservadas para avaliação cega de generalização (gen_01 a gen_10)
│   ├── anotacoes/              # Anotações manuais Ground Truth (.json exportados pelo Labelme)
│   └── mapeamento_imagens.json # Rastreabilidade das imagens originais da câmera
├── docs/                       # Especificações do edital, relatório preliminar e plano de desenvolvimento
│   ├── Plano_de_Desenvolvimento_PDI.pdf
│   ├── Projeto_da_Unidade_II.pdf
│   └── Relatório PDI.pdf
├── saidas/                     # Saídas automáticas do script (imagens processadas e CSVs)
├── processar.py                # Script CLI principal de processamento (atende as 4 saídas do edital)
├── avaliar_metricas.py         # Script de benchmark e comparação com Ground Truth
├── .gitignore
└── README.md
```

---

## ⚙️ Instalação e Requisitos

Clone o repositório e instale as dependências necessárias:

```bash
pip install opencv-python numpy matplotlib pandas reportlab labelme
```

---

## 🏷️ Anotação de Ground Truth (Labelme)

Consulte o guia completo em [docs/Plano_de_Desenvolvimento_PDI.pdf](docs/Plano_de_Desenvolvimento_PDI.pdf).

1. Abra o Labelme no terminal:
   ```bash
   labelme
   ```
2. Abra a pasta `dados/imagens/desenvolvimento/` ou `dados/imagens/generalizacao/`.
3. Configure `File -> Save Automatically` e marque como pasta de saída `dados/anotacoes/`.
4. Use o atalho **`Ctrl + N`** para contornar cada feijão com **Polígonos** (não use retângulos).
5. Nome da classe: digite **`semente`** (tudo minúsculo).

---

## 🚀 Como Executar o Pipeline de Processamento

Para processar uma imagem e gerar automaticamente as 4 saídas estritas do edital:

```bash
python processar.py dados/imagens/desenvolvimento/dev_01.jpg
```

### Saídas Geradas:
1. **Terminal:** Exibe `Total de sementes identificadas: {n}`.
2. **`saidas/contornos_*.png`:** Imagem com as Bounding Boxes (verde) e contornos individuais (vermelho).
3. **`saidas/fundo_preto_*.png`:** Imagem com os pixels originais dos feijões sobre fundo 100% preto.
4. **`saidas/metricas_*.csv`:** Tabela com coordenadas (`xmin`, `ymin`, `xmax`, `ymax`), `area`, `perimetro` e `circularidade`.

---

## 📊 Avaliação de Métricas (Benchmark)

Após salvar os arquivos `.json` do Labelme em `dados/anotacoes/`, execute o script de benchmark:

```bash
python avaliar_metricas.py
```

O script calcula automaticamente:
- **MAE de Contagem** ($|N_{\text{pred}} - N_{\text{real}}|$)
- **IoU de Bounding Boxes** com limiar $\ge 0.5$ (Precisão, Recall e F1-Score)
- **Segmentação Pixel a Pixel** (Índices de Jaccard e Dice)
- Relatório consolidado em `saidas/relatorio_benchmark.csv`.

