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
├── src/                        # Código-fonte Python (pipeline de PDI e scripts CLI)
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
