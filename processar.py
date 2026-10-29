

import argparse
import csv
import os
import sys
import cv2
import numpy as np


def processar_imagem(caminho_imagem, pasta_saida="saidas"):

    if not os.path.exists(caminho_imagem):
        print(f"Erro: O arquivo de imagem '{caminho_imagem}' não foi encontrado.")
        sys.exit(1)

    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"Erro: Falha ao carregar a imagem '{caminho_imagem}'. Verifique o formato do arquivo.")
        sys.exit(1)

    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

    # 1. ESPAÇO DE COR (HSV - Canal H / Matiz)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canal_h = hsv[:, :, 0]

    # 2. SUAVIZAÇÃO (Filtro de Mediana 15x15)
    canal_suavizado = cv2.medianBlur(canal_h, 15)

    # 3. LIMIARIZAÇÃO AUTOMÁTICA DE OTSU
    # Minimiza a variância intra-classe em histograma bimodal.
    limiar_calculado, mascara_binaria = cv2.threshold(
        canal_suavizado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. MORFOLOGIA MATEMÁTICA (Elemento Estruturante Elíptico 7x7)
    # - Fechamento (Dilatação -> Erosão, iter=2): Tampa fendas e poros da casca.
    # - Abertura (Erosão -> Dilatação, iter=1): Apaga ruídos pontuais no fundo.
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_fechada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel_morf, iterations=2)
    mascara_limpa = cv2.morphologyEx(mascara_fechada, cv2.MORPH_OPEN, kernel_morf, iterations=1)

    # 5. INDIVIDUALIZAÇÃO DE CONTATOS (Distância Euclidiana L2 + Watershed)
    contornos_preliminares, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marcador_fg = np.zeros(mascara_limpa.shape, dtype=np.uint8)

    for c in contornos_preliminares:
        area_comp = cv2.contourArea(c)
        if area_comp < 3000:  # Descarte de artefatos espúrios
            continue

        mascara_comp = np.zeros(mascara_limpa.shape, dtype=np.uint8)
        cv2.drawContours(mascara_comp, [c], -1, 255, -1)
        dist_comp = cv2.distanceTransform(mascara_comp, cv2.DIST_L2, 5)

        if area_comp < 18000:
            # Grão isolado: extrai o núcleo seguro (sure foreground)
            _, fg_local = cv2.threshold(dist_comp, 0.45 * dist_comp.max(), 255, 0)
            fg_local = np.uint8(fg_local)

            # Fallback por centróide geométrico via momentos caso o pico falhe
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
                _, fg_padrao = cv2.threshold(dist_comp, 0.75 * dist_comp.max(), 255, 0)
                marcador_fg = cv2.bitwise_or(marcador_fg, np.uint8(fg_padrao))

    # Rotulação de sementes confirmadas (Foreground)
    num_marcadores, marcadores = cv2.connectedComponents(marcador_fg)

    # Definição topológica do Watershed:
    # 0 = Linha de contato / Crista desconhecida (onde as bacias se encontram)
    # 1 = Fundo confirmado (EVA verde)
    # >= 2 = Sementes individualizadas
    fundo_seguro = cv2.dilate(mascara_limpa, kernel_morf, iterations=3)
    regiao_desconhecida = cv2.subtract(fundo_seguro, marcador_fg)

    marcadores = marcadores + 1
    marcadores[regiao_desconhecida == 255] = 0

    # Watershed: Inunda a partir dos marcadores e ergue cristas divisórias nos contatos
    marcadores = cv2.watershed(img, marcadores)


    imagem_contornos = img.copy()
    imagem_fundo_preto = np.zeros_like(img)
    lista_metricas = []

    for label in range(2, num_marcadores + 2):
        mascara_objeto = np.uint8(marcadores == label) * 255
        cnts_objeto, _ = cv2.findContours(mascara_objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_objeto:
            continue

        c_maior = max(cnts_objeto, key=cv2.contourArea)

        # Área (A): Contagem de pixels do contorno
        area = float(cv2.contourArea(c_maior))

        # Descarte de ruídos menores que 2500 pixels
        if area < 2500:
            continue

        # Perímetro (P): Comprimento da curva contínua fechada (pixels)
        perimetro = float(cv2.arcLength(c_maior, closed=True))

        # Circularidade Adimensional:
        # C = (4 * pi * A) / (P^2)
        # Teoria: Círculo perfeito = 1.0 | Quadrado = pi/4 (~0.785)
        # Feijões reais são elipsoidais/alongados -> valores típicos ~0.65 a 0.75
        circularidade = float((4 * np.pi * area) / (perimetro ** 2)) if perimetro > 0 else 0.0

        # Bounding Box: Menor retângulo envolvente [xmin, ymin, xmax, ymax]
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

        # Saída 2: Bounding Box (Verde) e Contorno Individualizado (Vermelho)
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

        # Saída 3: Pixels originais sobre fundo 100% preto
        imagem_fundo_preto[marcadores == label] = img[marcadores == label]

    caminho_contornos = os.path.join(pasta_saida, f"contornos_{nome_base}.png")
    caminho_fundo_preto = os.path.join(pasta_saida, f"fundo_preto_{nome_base}.png")
    caminho_csv = os.path.join(pasta_saida, f"metricas_{nome_base}.csv")

    cv2.imwrite(caminho_contornos, imagem_contornos)
    cv2.imwrite(caminho_fundo_preto, imagem_fundo_preto)

    # Saída 4: Exportação tabular CSV
    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as f:
        cabecalhos = ["id", "xmin", "ymin", "xmax", "ymax", "area", "perimetro", "circularidade"]
        writer = csv.DictWriter(f, fieldnames=cabecalhos)
        writer.writeheader()
        writer.writerows(lista_metricas)

    # Saída 1: Contagem impressa no console
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
        help="Caminho de uma imagem ou pasta (ex: dados/imagens/desenvolvimento/dev_01.jpg ou dados/imagens/desenvolvimento/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saidas",
        help="Diretório onde as saídas serão salvas (padrão: saidas/)"
    )

    args = parser.parse_args()

    # Suporte a processamento individual ou em lote
    if os.path.isdir(args.imagem):
        extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        arquivos = sorted([
            os.path.join(args.imagem, f) for f in os.listdir(args.imagem)
            if f.lower().endswith(extensoes)
        ])
        if not arquivos:
            print(f"Aviso: Nenhuma imagem encontrada no diretório '{args.imagem}'.")
            return

        print(f"=== Processando lote de {len(arquivos)} imagens em '{args.imagem}' ===")
        for arq in arquivos:
            print(f"\n--- Processando: {os.path.basename(arq)} ---")
            processar_imagem(arq, args.output_dir)
        print("\n=== Lote finalizado com sucesso! ===")
    else:
        processar_imagem(args.imagem, args.output_dir)


if __name__ == "__main__":
    main()
