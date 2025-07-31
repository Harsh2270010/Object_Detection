from PIL import Image
import numpy as np
import cv2

def draw_boxes(image, boxes, class_ids, class_names, scores=None):
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        if scores is not None:
            label = f"{class_names[class_id]}: {scores[i]:.2f}"
        else:
            label = class_names[class_id]
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
