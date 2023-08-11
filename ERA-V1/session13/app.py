import gradio as gr
import numpy as np
from config import PASCAL_CLASSES
from detect import predict


def inference(
    org_img: np.ndarray,
    iou_thresh: float,
    thresh: float,
    show_cam: str,
    transparency: float,
):
    outputs = predict(org_img, iou_thresh, thresh, show_cam, transparency)
    return outputs


title = "YoloV3 from Scratch on Pascal VOC Dataset with GradCAM"
description = f"Pytorch Implemetation of YoloV3 trained from scratch on Pascal VOC dataset with GradCAM \n Class in pascol voc: {', '.join(PASCAL_CLASSES)}"
examples = [
    ["images/000014.jpg", 0.5, 0.4, True, 0.5],
    ["images/000017.jpg", 0.6, 0.5, True, 0.5],
    ["images/000018.jpg", 0.55, 0.45, True, 0.5],
    ["images/000030.jpg", 0.5, 0.4, True, 0.5],
    ["images/Puppies.jpg", 0.6, 0.7, True, 0.5],
]

demo = gr.Interface(
    inference,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(0, 1, value=0.5, label="IOU Threshold"),
        gr.Slider(0, 1, value=0.4, label="Threshold"),
        gr.Checkbox(label="Show Grad Cam"),
        gr.Slider(0, 1, value=0.5, label="Opacity of GradCAM"),
    ],
    outputs=[
        gr.Gallery(rows=2, columns=1),
    ],
    title=title,
    description=description,
    examples=examples,
)
demo.launch()
