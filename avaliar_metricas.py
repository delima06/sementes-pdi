
import argparse
import csv
import glob
import json
import os
import sys
import cv2
import numpy as np

from processar import processar_imagem


def calcular_iou_bbox(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Interseção: largura e altura sobrepostas
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # União = Area1 + Area2 - Interseção
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    uniao = area1 + area2 - inter_area

    return (inter_area / uniao) if uniao > 0 else 0.0


def carregar_ground_truth_labelme(caminho_json):
    """
    Extrai do arquivo JSON do Labelme:
    1. Bounding Boxes de referência: [xmin, ymin, xmax, ymax]
    2. Máscara binária exata dos polígonos manuais (Ground Truth)
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    img_h = dados.get("imageHeight", 3072)
    img_w = dados.get("imageWidth", 3072)

    mascara_gt = np.zeros((img_h, img_w), dtype=np.uint8)
    bboxes_gt = []

    for shape in dados.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if "semente" in label or "feijao" in label or "grao" in label:
            pontos = np.array(shape.get("points"), dtype=np.int32)
            if len(pontos) >= 3:
                # Preenche polígono para formar a máscara de verdade de solo
                cv2.fillPoly(mascara_gt, [pontos], 255)
                xmin = int(np.min(pontos[:, 0]))
                ymin = int(np.min(pontos[:, 1]))
                xmax = int(np.max(pontos[:, 0]))
                ymax = int(np.max(pontos[:, 1]))
                bboxes_gt.append([xmin, ymin, xmax, ymax])

    return bboxes_gt, mascara_gt


def avaliar_imagem(caminho_imagem, caminho_json, pasta_saida="saidas"):
    """
    Calcula as métricas quantitativas de uma única imagem contra o Ground Truth:
    1. MAE de Contagem
    2. Correspondência de Bounding Boxes (TP, FP, FN, Precisão, Recall, F1)
    3. Segmentação Pixel a Pixel (Jaccard e Dice)
    """
    bboxes_gt, mask_gt = carregar_ground_truth_labelme(caminho_json)
    qtd_gt = len(bboxes_gt)

    total_pred, metricas_pred = processar_imagem(caminho_imagem, pasta_saida=pasta_saida)
    bboxes_pred = [[m["xmin"], m["ymin"], m["xmax"], m["ymax"]] for m in metricas_pred]

    # Carrega máscara predita do fundo preto gerado
    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]
    caminho_fp = os.path.join(pasta_saida, f"fundo_preto_{nome_base}.png")
    img_fp = cv2.imread(caminho_fp, cv2.IMREAD_GRAYSCALE)
    mask_pred = np.uint8(img_fp > 0) * 255 if img_fp is not None else np.zeros_like(mask_gt)

    # -------------------------------------------------------------------------
    # MÉTRICA 1: ERRO ABSOLUTO DE CONTAGEM (MAE)
    # Fórmula: Erro = |N_predito - N_real|
    # Avalia a precisão numérica global do algoritmo.
    # -------------------------------------------------------------------------
    erro_contagem = abs(total_pred - qtd_gt)

    # -------------------------------------------------------------------------
    # MÉTRICA 2: MATCHING DE BOUNDING BOXES (IoU >= 0.50)
    # Critério canônico do edital e literatura (PASCAL VOC):
    # - True Positive (TP): Caixa predita com IoU >= 0.50 contra caixa GT
    # - False Positive (FP): Caixa predita sem correspondência válida (falso alarme)
    # - False Negative (FN): Caixa GT não detectada pelo algoritmo
    # -------------------------------------------------------------------------
    matched_gt = set()
    ious_tp = []
    tp = 0

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

        if melhor_iou >= 0.50 and melhor_idx_gt != -1:
            tp += 1
            matched_gt.add(melhor_idx_gt)
            ious_tp.append(melhor_iou)

    fp = total_pred - tp
    fn = qtd_gt - tp

    # Fórmulas estatísticas clássicas:
    # Precisão = TP / (TP + FP)  -> De tudo que detectou, quanto era semente real?
    # Recall = TP / (TP + FN)    -> De todas as sementes reais, quantas achou?
    # F1-Score = 2 * (P * R) / (P + R) -> Média harmônica
    precisao = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) > 0 else 0.0
    iou_medio_bbox = float(np.mean(ious_tp)) if ious_tp else 0.0

    # -------------------------------------------------------------------------
    # MÉTRICA 3: SEGMENTAÇÃO PIXEL A PIXEL (Jaccard e Dice)
    # Avalia a qualidade do recorte geométrico em nível de pixel:
    # - Índice de Jaccard (IoU de Pixels) = |M_pred ∩ M_gt| / |M_pred ∪ M_gt|
    # - Coeficiente de Dice = (2 * |M_pred ∩ M_gt|) / (|M_pred| + |M_gt|)
    # -------------------------------------------------------------------------
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
    Varre a pasta de anotações manuais, executa o benchmark em todas as imagens
    disponíveis e consolida a tabela de métricas comparativas em CSV.
    """
    arquivos_json = sorted(glob.glob(os.path.join(pasta_anotacoes, "*.json")))
    if not arquivos_json:
        print(f"Aviso: Nenhum arquivo .json do Labelme encontrado em '{pasta_anotacoes}'.")
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
