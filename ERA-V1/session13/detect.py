from typing import List
import cv2
import torch
import numpy as np
import config
from pytorch_grad_cam.utils.image import show_cam_on_image

from yolov3 import YOLOv3
from utils import cells_to_bboxes, non_max_suppression, draw_predictions, YoloCAM


model = YOLOv3(num_classes=20)

model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()
print("[x] Model Loaded..")

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

cam = YoloCAM(model=model, target_layers=[model.layers[-2]], use_cuda=False)


@torch.inference_mode()
def predict(image: np.ndarray, iou_thresh: float = 0.5, thresh: float = 0.4, show_cam: bool = False, transparency: float = 0.5) -> List[np.ndarray]:
    transformed_image = config.transforms(image=image)["image"].unsqueeze(0)
    output = model(transformed_image)
    
    bboxes = [[] for _ in range(1)]
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = scaled_anchors[i]
        boxes_scale_i = cells_to_bboxes(
            output[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
    )
    plot_img = draw_predictions(image.copy(), nms_boxes, class_labels=config.PASCAL_CLASSES)
    if not show_cam:
        return [plot_img]
    
    grayscale_cam = cam(transformed_image, scaled_anchors)[0, :, :]
    img = cv2.resize(image, (416, 416))
    img = np.float32(img) / 255
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=transparency)
    return [plot_img, cam_image]


if __name__=="__main__":
    image = cv2.imread("images/Puppies.jpg")
    image = predict(image)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
