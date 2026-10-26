#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto de Processamento Digital de Imagens (PDI) - Unidade II
TAD0018 - Tecnologia em Análise e Desenvolvimento de Sistemas (EAJ / UFRN)

Script: avaliar_metricas.py
Objetivo:
    Avaliação quantitativa e validação científica do método computacional,
    comparando os resultados preditos com a Verdade de Solo (Ground Truth)
    anotada no software Labelme.

Alinhamento com o Relatório Técnico (docs/Relatório PDI.pdf):
    - Seção 2.2: Avaliação da generalização (Desenvolvimento vs Generalização).
    - Seção 2.4, Item Contagem: Erro Absoluto de Contagem (MAE): |N_pred - N_real|.
    - Seção 2.4, Item Identificação e Localização: Taxa de sobreposição (IoU de Bounding Boxes >= 0.5)
      com cálculo de True Positives (TP), False Positives (FP), False Negatives (FN), Precisão, Recall e F1.
    - Seção 2.4, Item Segmentação: Taxa de acerto pixel a pixel (Índice de Jaccard e Coeficiente de Dice).
    - Seção 2.4, Item Procedimentos Estatísticos: Média, Desvio Padrão e Coeficiente de Variação (CV).
"""

import argparse
import csv
import glob
import json
import os
import sys
import cv2
import numpy as np

# Importa o pipeline central diretamente
from processar import processar_imagem


def calcular_iou_bbox(box1, box2):
    """
    [Relatório - Seção 2.4: Taxa de Sobreposição / IoU de Bounding Boxes]
    Calcula o índice de Interseção sobre a União (Intersection over Union) entre duas caixas.
    IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)
    Formato esperado: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    uniao = area1 + area2 - inter_area
    return (inter_area / uniao) if uniao > 0 else 0.0


def carregar_ground_truth_labelme(caminho_json):
    """
    [Relatório - Seção 2.2: Referências Manuais do Labelme]
    Lê o arquivo de anotação .json exportado pelo Labelme e extrai:
    - Bounding Boxes de referência: [xmin, ymin, xmax, ymax]
    - Máscara binária exata dos polígonos manuais (Ground Truth)
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    img_h = dados.get("imageHeight", 3072)
    img_w = dados.get("imageWidth", 3072)

    mascara_gt = np.zeros((img_h, img_w), dtype=np.uint8)
    bboxes_gt = []

    for shape in dados.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        # Filtra a classe padronizada 'semente' definida no guia do grupo
        if "semente" in label or "feijao" in label or "grao" in label:
            pontos = np.array(shape.get("points"), dtype=np.int32)
            if len(pontos) >= 3:
                cv2.fillPoly(mascara_gt, [pontos], 255)
                xmin = int(np.min(pontos[:, 0]))
                ymin = int(np.min(pontos[:, 1]))
                xmax = int(np.max(pontos[:, 0]))
                ymax = int(np.max(pontos[:, 1]))
                bboxes_gt.append([xmin, ymin, xmax, ymax])

    return bboxes_gt, mascara_gt


def avaliar_imagem(caminho_imagem, caminho_json, pasta_saida="saidas"):
    """
    Compara a predição gerada pelo pipeline com o Ground Truth do Labelme.
    """
    bboxes_gt, mask_gt = carregar_ground_truth_labelme(caminho_json)
    qtd_gt = len(bboxes_gt)

    # Executa o algoritmo na imagem de entrada
    total_pred, metricas_pred = processar_imagem(caminho_imagem, pasta_saida=pasta_saida)
    bboxes_pred = [[m["xmin"], m["ymin"], m["xmax"], m["ymax"]] for m in metricas_pred]

    # Carrega a máscara binária predita a partir da imagem de fundo preto gerada
    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
    caminho_fp = os.path.join(pasta_saida, f"fundo_preto_{nome_base}.png")
    img_fp = cv2.imread(caminho_fp, cv2.IMREAD_GRAYSCALE)
    mask_pred = np.uint8(img_fp > 0) * 255 if img_fp is not None else np.zeros_like(mask_gt)

    # =========================================================================
    # [Relatório - Seção 2.4, Item 1: Erro Absoluto de Contagem (MAE)]
    # "Usaremos o Erro Absoluto Médio, o que é basicamente a diferença média
    #  entre a quantidade real de feijões e a quantidade que o programa contou:
    #  Erro Absoluto = |N_pred - N_real|"
    # =========================================================================
    erro_contagem = abs(total_pred - qtd_gt)

    # =========================================================================
    # [Relatório - Seção 2.4, Item 2: Identificação, Localização e Bounding Boxes]
    # "Para avaliar as bounding boxes, vamos calcular a taxa de sobreposição...
    #  Uma detecção é considerada Acerto (True Positive) se IoU >= 0.5."
    # =========================================================================
    matched_gt = set()
    ious_tp = []
    tp = 0

    # Correspondência gulosa por maior IoU
    for b_pred in bboxes_pred:
        melhor_iou = 0.0
        melhor_idx_gt = -1

        for idx_gt, b_gt in enumerate(bboxes_gt):
            if idx_gt in matched_gt:
                continue
            iou = calcular_iou_bbox(b_pred, b_gt)
            if iou > melhor_iou:
                melhor_iou = iou
                melhor_idx_gt = idx_gt

        # Limiar de acerto padronizado no edital e literatura (IoU >= 0.50)
        if melhor_iou >= 0.5 and melhor_idx_gt != -1:
            tp += 1
            matched_gt.add(melhor_idx_gt)
            ious_tp.append(melhor_iou)

    fp = total_pred - tp  # Predições sem correspondência (falsos positivos)
    fn = qtd_gt - tp      # Sementes reais não detectadas (falsos negativos)

    precisao = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) > 0 else 0.0
    iou_medio_bbox = float(np.mean(ious_tp)) if ious_tp else 0.0

    # =========================================================================
    # [Relatório - Seção 2.4, Item 3: Segmentação Pixel a Pixel (Jaccard e Dice)]
    # "Para avaliar os pixels da semente em si, vamos medir a taxa de acerto de
    #  pixels, comparando se a região limiarizada pelo nosso método bate com o
    #  polígono perfeito do Labelme."
    #  - Jaccard (IoU de Pixels) = |M_pred ∩ M_gt| / |M_pred ∪ M_gt|
    #  - Dice Score = 2 * |M_pred ∩ M_gt| / (|M_pred| + |M_gt|)
    # =========================================================================
    intersecao_pixel = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    uniao_pixel = np.logical_or(mask_pred > 0, mask_gt > 0).sum()
    soma_pixels = (mask_pred > 0).sum() + (mask_gt > 0).sum()

    iou_pixel = float(intersecao_pixel / uniao_pixel) if uniao_pixel > 0 else 0.0
    dice_pixel = float((2.0 * intersecao_pixel) / soma_pixels) if soma_pixels > 0 else 0.0

    return {
        "imagem": os.path.basename(caminho_imagem),
        "qtd_real": qtd_gt,
        "qtd_predita": total_pred,
        "mae_contagem": erro_contagem,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precisao_bbox": round(precisao, 4),
        "recall_bbox": round(recall, 4),
        "f1_bbox": round(f1, 4),
        "iou_medio_bbox": round(iou_medio_bbox, 4),
        "iou_pixels_jaccard": round(iou_pixel, 4),
        "dice_pixels": round(dice_pixel, 4),
    }


def executar_avaliacao_geral(pasta_anotacoes="dados/anotacoes", pasta_saida="saidas"):
    """
    [Relatório - Seção 2.4: Procedimentos Estatísticos e Generalização]
    Varre a pasta de anotações manuais, executa o benchmark em todas as imagens
    disponíveis e consolida a tabela de métricas comparativas em CSV.
    """
    arquivos_json = sorted(glob.glob(os.path.join(pasta_anotacoes, "*.json")))
    if not arquivos_json:
        print(f"Aviso: Nenhum arquivo .json do Labelme encontrado em '{pasta_anotacoes}'.")
        print("Assim que as anotações forem salvas nessa pasta, execute este script para gerar o benchmark.")
        return

    print(f"Encontrados {len(arquivos_json)} arquivos de anotação Ground Truth.\nIniciando avaliação...")

    resultados = []
    for arq_json in arquivos_json:
        nome_base = os.path.splitext(os.path.basename(arq_json))[0]

        # Busca a imagem correspondente em desenvolvimento ou generalizacao
        caminho_img = None
        for subpasta in ["desenvolvimento", "generalizacao"]:
            for ext in [".jpg", ".jpeg", ".png"]:
                teste_caminho = os.path.join("dados", "imagens", subpasta, nome_base + ext)
                if os.path.exists(teste_caminho):
                    caminho_img = teste_caminho
                    break
            if caminho_img:
                break

        if not caminho_img:
            print(f"Aviso: Imagem correspondente para '{arq_json}' não encontrada. Pulando...")
            continue

        res = avaliar_imagem(caminho_img, arq_json, pasta_saida=pasta_saida)
        resultados.append(res)
        print(
            f"-> {res['imagem']}: Real={res['qtd_real']} | Pred={res['qtd_predita']} | "
            f"MAE={res['mae_contagem']} | IoU_Pixel={res['iou_pixels_jaccard']} | F1_BBox={res['f1_bbox']}"
        )

    if resultados:
        caminho_relatorio = os.path.join(pasta_saida, "relatorio_benchmark.csv")
        with open(caminho_relatorio, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\nRelatório de benchmark consolidado salvo em: {caminho_relatorio}")


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação de métricas e benchmark com Ground Truth (Labelme)"
    )
    parser.add_argument(
        "--anotacoes",
        type=str,
        default="dados/anotacoes",
        help="Diretório com os arquivos .json gerados pelo Labelme (padrão: dados/anotacoes)"
    )
    parser.add_argument(
        "--saidas",
        type=str,
        default="saidas",
        help="Diretório de saídas do algoritmo (padrão: saidas/)"
    )

    args = parser.parse_args()
    executar_avaliacao_geral(pasta_anotacoes=args.anotacoes, pasta_saida=args.saidas)


if __name__ == "__main__":
    main()
