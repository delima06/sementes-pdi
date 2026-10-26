#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto de Processamento Digital de Imagens (PDI) - Unidade II
TAD0018 - Tecnologia em Análise e Desenvolvimento de Sistemas (EAJ / UFRN)

Script: processar.py
Objetivo:
    Identificação, segmentação, contagem automática e extração de características
    físicas de sementes (grãos de feijão) dispostas sobre superfície de EVA verde.

Alinhamento com o Relatório Técnico (docs/Relatório PDI.pdf):
    - Seção 2.1: Aquisição padronizada a 90 graus para evitar distorções de escala.
    - Seção 2.3, Item 1: Conversão para espaço de cor HSV (separação cor vs luminância).
    - Seção 2.3, Item 2: Filtragem de Mediana para remoção de ruídos preservando bordas.
    - Seção 2.3, Item 3: Limiarização automática de Otsu em histograma bimodal.
    - Seção 2.3, Item 4: Operações morfológicas de Fechamento e Abertura.
    - Seção 2.3, Item 5: Extração morfológica (Área, Perímetro, Circularidade e BBoxes).
    - Edital / Plano: Individualização de feijões em contato via Distância Euclidiana e Watershed.

4 Saídas Estritas do Edital:
    1. Contagem impressa no console (Total de sementes identificadas: {n})
    2. Imagem com Bounding Boxes e contornos desenhados (saidas/contornos_*.png)
    3. Imagem com as sementes em seus pixels originais sobre fundo 100% preto (saidas/fundo_preto_*.png)
    4. Arquivo CSV tabular com xmin, ymin, xmax, ymax, área, perímetro e circularidade (saidas/metricas_*.csv)
"""

import argparse
import csv
import os
import sys
import cv2
import numpy as np


def processar_imagem(caminho_imagem, pasta_saida="saidas"):
    """
    Executa o pipeline completo de Processamento Digital de Imagens sobre uma imagem de entrada.
    """
    if not os.path.exists(caminho_imagem):
        print(f"Erro: O arquivo de imagem '{caminho_imagem}' não foi encontrado.")
        sys.exit(1)

    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"Erro: Falha ao carregar a imagem '{caminho_imagem}'. Verifique o formato do arquivo.")
        sys.exit(1)

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

    # =========================================================================
    # [Relatório - Seção 2.3, Item 1: Conversão de Espaço de Cor]
    # "Convertemos a imagem colorida original para o formato HSV... porque o
    #  contraste entre o EVA verde e os feijões fica muito mais evidente nos canais
    #  de cor do que na simples intensidade de luz, facilitando a separação de fundo."
    # =========================================================================
    # O canal H (Hue / Matiz) codifica o tom de cor. O fundo verde de EVA possui H
    # elevado (~75-85 no OpenCV), enquanto os grãos de feijão possuem H significativamente
    # mais baixo (< 45), gerando um contraste extremo independente de variações de brilho.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canal_h = hsv[:, :, 0]

    # =========================================================================
    # [Relatório - Seção 2.3, Item 2: Filtragem de Suavização (Restauração)]
    # "Como as imagens de câmeras podem ter pequenos ruídos, decidimos aplicar o
    #  Filtro de Mediana, sendo utilizado para remover ruídos isolados (como o ruído
    #  sal e pimenta) e tem a grande vantagem de preservar a nitidez das bordas
    #  dos feijões, diferente de um filtro de média comum que deixaria tudo borrado."
    # =========================================================================
    # Aplicamos a mediana com kernel 15x15 para atenuar a textura porosa do EVA e
    # ruídos de sensor da câmera do celular, preservando o contorno exato da casca.
    canal_suavizado = cv2.medianBlur(canal_h, 15)

    # =========================================================================
    # [Relatório - Seção 2.3, Item 3: Limiarização de Otsu]
    # "O método de Otsu é ideal quando o histograma da imagem é bimodal (dois picos
    #  bem definidos). Como o fundo verde e os feijões formam duas classes de cores
    #  muito distintas nesse canal, o algoritmo calcula automaticamente o melhor
    #  limiar (T) que separa os feijões do fundo."
    # =========================================================================
    # Usamos THRESH_BINARY_INV porque os feijões têm H menor (ficam brancos = 255)
    # e o fundo verde tem H maior (fica preto = 0).
    limiar_calculado, mascara_binaria = cv2.threshold(
        canal_suavizado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # =========================================================================
    # [Relatório - Seção 2.3, Item 4: Operações Morfológicas]
    # "Aplicamos métodos morfológicos básicos de Fechamento (para fechar buracos
    #  dentro do feijão) e Abertura (para remover pontinhos do fundo) e ajudar a
    #  desconectar feijões que ficaram levemente na imagem binária."
    # =========================================================================
    # Elemento estruturante elíptico (7x7) para respeitar o formato anatômico das sementes.
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Fechamento (Dilatação seguida de Erosão): preenche eventuais sombras e poros da casca
    mascara_fechada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel_morf, iterations=2)

    # Abertura (Erosão seguida de Dilatação): elimina ruídos isolados residuais no fundo de EVA
    mascara_limpa = cv2.morphologyEx(mascara_fechada, cv2.MORPH_OPEN, kernel_morf, iterations=1)

    # =========================================================================
    # [Edital / Plano de Desenvolvimento: Individualização de Feijões em Contato]
    # "Individualização de feijões em contato: aplicar Transformada de Distância e
    #  Watershed. Uma semente constitui um objeto independente, inclusive quando
    #  estiver em contato com outras."
    # =========================================================================
    # Para separar grãos que se tocam, analisamos cada componente conectado da máscara:
    # 1. Grãos isolados (área < 18.000 pixels): threshold padrão na distância euclidiana.
    # 2. Aglomerados em contato (área >= 18.000 pixels): aplicamos limiarização adaptativa
    #    progressiva para identificar com precisão os múltiplos centros de gravidade (picos).
    contornos_preliminares, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marcador_fg = np.zeros(mascara_limpa.shape, dtype=np.uint8)

    for c in contornos_preliminares:
        area_comp = cv2.contourArea(c)
        # Descarte de pequenos artefatos espúrios (< 3000 px)
        if area_comp < 3000:
            continue

        mascara_comp = np.zeros(mascara_limpa.shape, dtype=np.uint8)
        cv2.drawContours(mascara_comp, [c], -1, 255, -1)
        dist_comp = cv2.distanceTransform(mascara_comp, cv2.DIST_L2, 5)

        if area_comp < 18000:
            # Grão isolado: extrai o núcleo seguro (sure foreground)
            _, fg_local = cv2.threshold(dist_comp, 0.45 * dist_comp.max(), 255, 0)
            fg_local = np.uint8(fg_local)

            if cv2.countNonZero(fg_local) == 0:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                    cv2.circle(marcador_fg, (cx, cy), 5, 255, -1)
            else:
                marcador_fg = cv2.bitwise_or(marcador_fg, fg_local)
        else:
            # Aglomerado com feijões em contato: varredura na distância até individualizar os núcleos
            fg_escolhido = None
            for ratio in np.arange(0.50, 0.75, 0.05):
                _, fg_teste = cv2.threshold(dist_comp, ratio * dist_comp.max(), 255, 0)
                fg_teste = np.uint8(fg_teste)
                qtd_blobs, _ = cv2.connectedComponents(fg_teste)
                qtd_blobs -= 1

                # Estimativa baseada no tamanho médio de cada feijão (~12.000 px)
                estimativa_feijoes = max(2, int(round(area_comp / 12000.0)))
                if qtd_blobs >= estimativa_feijoes or ratio >= 0.70:
                    fg_escolhido = fg_teste
                    break

            if fg_escolhido is not None and cv2.countNonZero(fg_escolhido) > 0:
                marcador_fg = cv2.bitwise_or(marcador_fg, fg_escolhido)
            else:
                _, fg_padrao = cv2.threshold(dist_comp, 0.60 * dist_comp.max(), 255, 0)
                marcador_fg = cv2.bitwise_or(marcador_fg, np.uint8(fg_padrao))

    # Rotulação dos componentes conectados dos feijões
    num_marcadores, marcadores = cv2.connectedComponents(marcador_fg)

    # Definição do fundo seguro e da zona de incerteza (área desconhecida)
    fundo_seguro = cv2.dilate(mascara_limpa, kernel_morf, iterations=3)
    regiao_desconhecida = cv2.subtract(fundo_seguro, marcador_fg)

    # Preparação dos marcadores para o algoritmo cv2.watershed:
    # 0 = zona desconhecida (onde as bacias hidrográficas se expandem e encontram cristas)
    # 1 = fundo confirmado (EVA verde)
    # >= 2 = sementes confirmadas
    marcadores = marcadores + 1
    marcadores[regiao_desconhecida == 255] = 0

    # Aplicação da bacia hidrográfica sobre a imagem original
    marcadores = cv2.watershed(img, marcadores)

    # =========================================================================
    # [Relatório - Seção 2.3, Item 5: Extração e Cálculos]
    # "Com os feijões isolados, o OpenCV localiza os contornos de cada um.
    #  A partir disso, contamos a quantidade total e calculamos a Área (A)
    #  contando os pixels, o Perímetro (P) seguindo a borda, e aplicamos a
    #  fórmula de Circularidade = (4*pi*A)/(P²). Também extraímos as coordenadas
    #  da bounding box de cada feijão."
    # =========================================================================
    imagem_contornos = img.copy()
    imagem_fundo_preto = np.zeros_like(img)
    lista_metricas = []

    for label in range(2, num_marcadores + 2):
        mascara_objeto = np.uint8(marcadores == label) * 255
        cnts_objeto, _ = cv2.findContours(mascara_objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_objeto:
            continue

        c_maior = max(cnts_objeto, key=cv2.contourArea)
        area = float(cv2.contourArea(c_maior))

        # Descarte de ruídos menores que 2500 pixels
        if area < 2500:
            continue

        perimetro = float(cv2.arcLength(c_maior, True))

        # Fórmula matemática de circularidade estipulada no edital:
        # C = (4 * pi * Area) / (Perimetro^2)
        # 1.0 representa um círculo perfeito; pi/4 (~0.785) representa um quadrado perfeito
        circularidade = float((4 * np.pi * area) / (perimetro ** 2)) if perimetro > 0 else 0.0

        # Bounding box delimitadora: menor retângulo que contém a semente
        x, y, w, h = cv2.boundingRect(c_maior)
        xmin, ymin, xmax, ymax = x, y, x + w, y + h

        semente_id = len(lista_metricas) + 1
        lista_metricas.append({
            "id": semente_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "area": round(area, 2),
            "perimetro": round(perimetro, 2),
            "circularidade": round(circularidade, 4)
        })

        # SAÍDA 2: Desenho dos retângulos envolventes (verde) e contornos individuais (vermelho)
        cv2.rectangle(imagem_contornos, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.drawContours(imagem_contornos, [c_maior], -1, (0, 0, 255), 2)
        cv2.putText(
            imagem_contornos,
            str(semente_id),
            (xmin, max(ymin - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        # SAÍDA 3: Sementes com seus pixels originais sobre fundo 100% preto
        imagem_fundo_preto[marcadores == label] = img[marcadores == label]

    # =========================================================================
    # Geração dos arquivos de saída obrigatórios
    # =========================================================================
    caminho_contornos = os.path.join(pasta_saida, f"contornos_{nome_base}.png")
    caminho_fundo_preto = os.path.join(pasta_saida, f"fundo_preto_{nome_base}.png")
    caminho_csv = os.path.join(pasta_saida, f"metricas_{nome_base}.csv")

    cv2.imwrite(caminho_contornos, imagem_contornos)
    cv2.imwrite(caminho_fundo_preto, imagem_fundo_preto)

    # SAÍDA 4: Arquivo CSV Tabular com as métricas individuais
    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as f:
        cabecalhos = ["id", "xmin", "ymin", "xmax", "ymax", "area", "perimetro", "circularidade"]
        writer = csv.DictWriter(f, fieldnames=cabecalhos)
        writer.writeheader()
        writer.writerows(lista_metricas)

    # SAÍDA 1: Exibição no console do número total de sementes identificadas
    print(f"Total de sementes identificadas: {len(lista_metricas)}")
    print(f"Imagem com contornos salva em: {caminho_contornos}")
    print(f"Imagem com fundo preto salva em: {caminho_fundo_preto}")
    print(f"Métricas CSV salvas em: {caminho_csv}")

    return len(lista_metricas), lista_metricas


def main():
    parser = argparse.ArgumentParser(
        description="Segmentação, identificação e contagem automática de sementes (TAD0018 - PDI)"
    )
    parser.add_argument(
        "imagem",
        type=str,
        help="Caminho do arquivo de imagem (ex: dados/imagens/desenvolvimento/dev_01.jpg)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saidas",
        help="Diretório onde as saídas serão salvas (padrão: saidas/)"
    )

    args = parser.parse_args()
    processar_imagem(args.imagem, args.output_dir)


if __name__ == "__main__":
    main()
