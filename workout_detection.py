import cv2
from ultralytics import YOLO

# ── Cargar el modelo ──────────────────────────────────────────────────────────
# Descarga best.pt desde Drive a tu computador primero
MODEL_PATH = "best.pt" 
model = YOLO(MODEL_PATH)

# ── Colores por clase ─────────────────────────────────────────────────────────
COLORS = [
    (255, 56,  56),  (255, 157, 51),  (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187)
]

# ── Abrir cámara ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Cámara abierta. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Inferencia ────────────────────────────────────────────────────────────
    results = model(frame, conf=0.35, verbose=False)[0]

    # ── Dibujar detecciones ───────────────────────────────────────────────────
    if results.boxes is not None:
        for box in results.boxes:
            cls_idx  = int(box.cls.item())
            conf     = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_name = results.names[cls_idx]
            color    = COLORS[cls_idx % len(COLORS)]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Etiqueta con fondo
            label = f"{cls_name.replace('_', ' ').title()}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Info en pantalla ──────────────────────────────────────────────────────
    n_det = len(results.boxes) if results.boxes is not None else 0
    cv2.putText(frame, f"Detecciones: {n_det}  |  ESC para salir",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Exercise Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()