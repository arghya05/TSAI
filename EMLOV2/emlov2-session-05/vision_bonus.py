import os
from datetime import datetime
from typing import List, Tuple

import boto3
import torch
import gradio as gr

from utils import CustomFlagging


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

    def recognize_cifar(im):
        if im is None:
            return None

        st = datetime.now()
        image = torch.tensor(im[None, ...], dtype=torch.float32)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        out = {classes[i]: preds[i] for i in range(10)}
        et = datetime.now()
        print(out)

        inf_time = (et - st).total_seconds() * 10 ** 3
        flag_callback.flag(
            im=im,
            output=out,
            flag_data={"timestamp": st, "infrence_time (ms)": inf_time},
        )
        flag_callback.push_to_s3(s3, OUTPUT_BUCKET, f"{OUTPUT_DIR}/{FLAGGED_DIR}")
        return out

    im = gr.Image(shape=(32, 32))

    demo = gr.Interface(
        fn=recognize_cifar,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
        allow_flagging="never",
    )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":

    _, _, MODEL_BUCKET, MODEL_DIR, MODEL_NAME = os.environ.get("model").split("/")
    _, _, OUTPUT_BUCKET, OUTPUT_DIR, FLAGGED_DIR = os.environ.get("flagged_dir").split(
        "/"
    )

    print(MODEL_BUCKET, MODEL_DIR, MODEL_NAME)
    print(OUTPUT_BUCKET, OUTPUT_DIR, FLAGGED_DIR)

    s3 = boto3.client("s3")
    if not os.path.exists(MODEL_NAME):
        print("[INFO] Downloading model from S3 Bucket..")
        s3.download_file(MODEL_BUCKET, f"{MODEL_DIR}/{MODEL_NAME}", MODEL_NAME)
        print("downloaded")
    else:
        print("[INFO] model exists..")

    flag_callback = CustomFlagging(flagging_dir="flagged_bonus")
    main(MODEL_NAME)
