"""
hand draw - Рисовалка с помощью жестов руки

Рисовалка: рисовать когда большой и указательный пальцы сведены.
Зависимости: mediapipe, opencv-python, numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import time

# ---------------------------
# Параметры и константы
# ---------------------------
CAMERA_ID = 0
MAX_NUM_HANDS = 1
DRAWING_COLOR_PRESETS = [
    (0, 0, 255),    # красный (BGR)
    (255, 0, 0),    # синий
    (0, 255, 0),    # зелёный
    (0, 255, 255),  # жёлтый (BGR)
]
ERASER_LABEL = "ERASE"
BUTTON_HOVER_FRAMES = 15  # сколько кадров держать палец над кнопкой для выбора
THRESHOLD_NORM = 0.25     # порог нормализованного расстояния между пальцами (настройка)

BRUSH_THICKNESS = 8
ERASER_THICKNESS = 50

# ---------------------------
# Инициализация mediapipe
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---------------------------
# Вспомогательные функции
# ---------------------------
def lm_to_px(lm, w, h):
    """Перевод нормализованных координат landmark в пиксели (x, y)."""
    return int(lm.x * w), int(lm.y * h)

def normalized_distance(lm1, lm2, ref_dist):
    """
    Возвращает нормализованное расстояние между двумя landmark (евклид),
    нормированное на ref_dist (чтобы порог был масштаб-независим).
    """
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return (dx * dx + dy * dy) ** 0.5 / max(ref_dist, 1e-6)

def draw_V_indicator(frame, p1, p2, color=(255, 0, 0), thickness=4):
    """
    Рисует простой 'V' (стрелка/контур) между двумя точками (p1, p2).
    p1, p2 — (x, y)
    """
    x1, y1 = p1
    x2, y2 = p2
    # вычислим центр и небольшой уголчик
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # вектор от thumb к index
    vx, vy = x2 - x1, y2 - y1
    # перпендикуляр
    px, py = -vy, vx
    # нормируем перпендикуляр
    norm = max((px * px + py * py) ** 0.5, 1e-6)
    px, py = int(px / norm * 30), int(py / norm * 30)
    # вершины V
    pt1 = (cx - px, cy - py)
    pt2 = (cx + px, cy + py)
    # рисуем V-образный контур
    cv2.line(frame, pt1, (cx, cy), color, thickness)
    cv2.line(frame, pt2, (cx, cy), color, thickness)
    # маленький кружок-цель
    cv2.circle(frame, (cx, cy), 6, color, 2)

# ---------------------------
# Основная функция
# ---------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры.")
        return

    h, w = frame.shape[:2]

    # canvas_color: хранит цвета нарисованных пикселей
    canvas_color = np.zeros_like(frame, dtype=np.uint8)
    # canvas_mask: где есть рисунок (255) или нет (0)
    canvas_mask = np.zeros((h, w), dtype=np.uint8)

    # текущее состояние палитры
    palette_buttons = []
    pad = 10
    rect_w = 70
    rect_h = 50
    start_x = pad
    start_y = pad

    # Создаём кнопки палитры: для каждого цвета + ластик
    for clr in DRAWING_COLOR_PRESETS:
        rect = (start_x, start_y, rect_w, rect_h)
        palette_buttons.append({
            "rect": rect,
            "color": clr,
            "label": None,
            "hover_count": 0,
        })
        start_x += rect_w + pad

    # Ластик (white square с этикеткой)
    rect = (start_x, start_y, rect_w, rect_h)
    palette_buttons.append({
        "rect": rect,
        "color": (255, 255, 255),
        "label": ERASER_LABEL,
        "hover_count": 0,
    })

    selected_color = DRAWING_COLOR_PRESETS[0]
    eraser_mode = False

    prev_x, prev_y = None, None
    drawing = False  # должны ли мы в этот момент рисовать (пальцы соединены)
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # зеркально для удобства
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h, w = frame.shape[:2]

        index_tip_px = None
        thumb_tip_px = None
        norm_ref = 0.1  # референс дистанции для нормализации (обновится если есть руки)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # первая рука
            # нормировочная дистанция: wrist(0) <-> middle_finger_mcp(9)
            ref_lm = hand_landmarks.landmark[0]
            ref_lm2 = hand_landmarks.landmark[9]
            norm_ref = ((ref_lm.x - ref_lm2.x) ** 2 + (ref_lm.y - ref_lm2.y) ** 2) ** 0.5

            lm_thumb = hand_landmarks.landmark[4]
            lm_index = hand_landmarks.landmark[8]
            thumb_tip_px = lm_to_px(lm_thumb, w, h)
            index_tip_px = lm_to_px(lm_index, w, h)

            # нормализованное расстояние между пальцами
            norm_dist = normalized_distance(lm_thumb, lm_index, norm_ref)

            # Определяем состояние: рисование включено если пальцы близко
            if norm_dist < THRESHOLD_NORM:
                # пальцы сведены -> рисование ON
                drawing = True
            else:
                drawing = False

            # Рисуем скелет руки (можно отключить, если мешает)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1))

        else:
            drawing = False

        # ---------------------
        # Обработка палитры: определяем наведение указательного пальца по rect-ам
        # ---------------------
        # Если есть координаты указательного — используем их для hover/выбора
        if index_tip_px:
            ix, iy = index_tip_px
            for btn in palette_buttons:
                x, y, rw, rh = btn["rect"]
                if x <= ix <= x + rw and y <= iy <= y + rh:
                    btn["hover_count"] += 1
                else:
                    btn["hover_count"] = 0

                # Если задержали палец в области достаточно долго — производим выбор
                if btn["hover_count"] >= BUTTON_HOVER_FRAMES:
                    # выбор только если палец не движется? здесь просто переключаем
                    if btn["label"] == ERASER_LABEL:
                        eraser_mode = True
                    else:
                        eraser_mode = False
                        selected_color = btn["color"]

                    # сбросим hover, чтобы снова нельзя было сразу переключить
                    btn["hover_count"] = 0

        # ---------------------
        # Рисование / Стирание на canvas
        # ---------------------
        if drawing and index_tip_px:
            # показать зелёный курсор
            cx, cy = index_tip_px
            cursor_color = (0, 255, 0)
            # если только что начали рисовать — установим prev равным текущему
            if prev_x is None or prev_y is None:
                prev_x, prev_y = cx, cy

            if eraser_mode:
                # стираем: обнуляем маску и цвет в области
                cv2.line(canvas_mask, (prev_x, prev_y), (cx, cy), 0, ERASER_THICKNESS)
                cv2.circle(canvas_mask, (cx, cy), ERASER_THICKNESS // 2, 0, -1)
                # также обнулим в canvas_color (чтобы цветовые остатки исчезли)
                # создаём маску для обнуления
                erase_mask = np.zeros_like(canvas_mask)
                cv2.line(erase_mask, (prev_x, prev_y), (cx, cy), 255, ERASER_THICKNESS)
                cv2.circle(erase_mask, (cx, cy), ERASER_THICKNESS // 2, 255, -1)
                canvas_color[erase_mask == 255] = 0
            else:
                # рисуем линию: обновляем canvas_color и canvas_mask
                color = selected_color
                # линия на цветном холсте
                cv2.line(canvas_color, (prev_x, prev_y), (cx, cy), color, BRUSH_THICKNESS)
                cv2.circle(canvas_color, (cx, cy), BRUSH_THICKNESS // 2, color, -1)
                # маска ставим 255 там, где нарисовали
                cv2.line(canvas_mask, (prev_x, prev_y), (cx, cy), 255, BRUSH_THICKNESS)
                cv2.circle(canvas_mask, (cx, cy), BRUSH_THICKNESS // 2, 255, -1)

            # рисуем курсор кружком
            cv2.circle(frame, (cx, cy), 8, cursor_color, -1)

            prev_x, prev_y = cx, cy
        else:
            # режим паузы: рисование прекращается
            # курсор красный если есть указательный, иначе ничего
            if index_tip_px:
                ix, iy = index_tip_px
                cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
            prev_x, prev_y = None, None

            # если пальцы разъединены и есть точки — показать V-индикатор между ними
            if thumb_tip_px and index_tip_px and not drawing:
                draw_V_indicator(frame, thumb_tip_px, index_tip_px, color=(255, 0, 0), thickness=3)

        # ---------------------
        # Наложение canvas на frame (используя mask)
        # ---------------------
        # где mask == 255 — берём пиксели из canvas_color, иначе из frame
        mask_3c = cv2.merge([canvas_mask, canvas_mask, canvas_mask])
        inv_mask = cv2.bitwise_not(mask_3c)
        frame_bg = cv2.bitwise_and(frame, inv_mask)
        drawing_fg = cv2.bitwise_and(canvas_color, mask_3c)
        overlayed = cv2.add(frame_bg, drawing_fg)

        # ---------------------
        # Рисуем палитру сверху
        # ---------------------
        for btn in palette_buttons:
            x, y, rw, rh = btn["rect"]
            clr = btn["color"]
            label = btn["label"]
            # фон кнопки (там где выбран — подчеркнём)
            cv2.rectangle(overlayed, (x, y), (x + rw, y + rh), (200, 200, 200), -1)
            # внутри — цветной прямоугольник (с отступом)
            inner = 6
            cv2.rectangle(overlayed, (x + inner, y + inner), (x + rw - inner, y + rh - inner), clr, -1)
            # рамка
            cv2.rectangle(overlayed, (x, y), (x + rw, y + rh), (100, 100, 100), 2)

            # Если кнопка выбрана — рисуем толстую рамку
            if (not eraser_mode and label is None and clr == selected_color) or (label == ERASER_LABEL and eraser_mode):
                cv2.rectangle(overlayed, (x - 3, y - 3), (x + rw + 3, y + rh + 3), (0, 255, 0), 3)

            # если label есть — отобразим буквы
            if label:
                cv2.putText(overlayed, label, (x + 8, y + rh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

            # показать прогресс hover (если есть)
            if btn["hover_count"] > 0:
                prog = int(btn["hover_count"] / BUTTON_HOVER_FRAMES * rw)
                cv2.rectangle(overlayed, (x, y + rh + 2), (x + prog, y + rh + 8), (50, 200, 50), -1)

        # подсказка управления
        cv2.putText(overlayed, "Hover on color to select | Hover on ERASE to erase | Press 'c' to clear | 'q' to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        cv2.imshow("Hand Draw (press q to quit)", overlayed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('c') or key == ord('r'):
            # очистка холста
            canvas_color[:] = 0
            canvas_mask[:] = 0
            print("Canvas cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()