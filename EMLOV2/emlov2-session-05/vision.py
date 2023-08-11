import os
from typing import List, Tuple

import boto3
import gradio as gr
import torch


def main(model_path: str) -> Tuple[dict, dict]:
    model = torch.jit.load(model_path)

    classes = (
        "plane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def recognize_cifar(image):
        if image is None:
            return None

        image = torch.tensor(image[None, ...], dtype=torch.float32)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        out = {classes[i]: preds[i] for i in range(10)}
        print(out)
        return out

    im = gr.Image(shape=(32, 32))

    demo = gr.Interface(
        fn=recognize_cifar,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
        allow_flagging="manual",
        # flagging_dir="inputs",
    )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    MODEL_NAME = "model.traced.pt"
    if not os.path.exists(MODEL_NAME):
        print("[INFO] Downloading model from S3 Bucket..")
        s3 = boto3.client("s3")
        s3.download_file("cifar10-emlov2", "model.traced.pt", MODEL_NAME)
        print("downloaded")
    else:
        print("[INFO] model exists..")

    main(MODEL_NAME)
