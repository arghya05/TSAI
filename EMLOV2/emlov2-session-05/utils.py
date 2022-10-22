import csv
import json
import os
import uuid
from typing import Dict, List

import cv2
import numpy as np


class CustomFlagging:
    def __init__(
        self,
        flagging_dir: str,
        fieldnames: List = [
            "image",
            "output",
            "username",
            "timestamp",
            "infrence_time (ms)",
        ],
    ):
        self.logs_path = os.path.join(flagging_dir, "log.csv")
        self.flagging_dir = flagging_dir
        self.fieldnames = fieldnames

        if not os.path.exists(flagging_dir):
            os.makedirs(flagging_dir)
            with open(self.logs_path, "w", encoding="UTF8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        os.makedirs(os.path.join(self.flagging_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(self.flagging_dir, "output"), exist_ok=True)

    def flag(self, im: np.array, output: Dict, flag_data: Dict):
        unique_filename = str(uuid.uuid4())
        im_path = os.path.join(self.flagging_dir, "image", f"{unique_filename}.jpg")
        out_path = os.path.join(self.flagging_dir, "output", f"{unique_filename}.json")

        flag_data["image"] = im_path
        flag_data["output"] = out_path

        cv2.imwrite(im_path, im)
        with open(out_path, "w") as f:
            f.write(json.dumps(output))

        with open(self.logs_path, "a", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows([flag_data])

    def push_to_s3(self, s3, upload_bucket: str, upload_path: str):
        with open(self.logs_path, "r") as f:
            for line in csv.DictReader(f):
                data = line

        s3.upload_file(
            data["image"], upload_bucket, os.path.join(upload_path, data["image"])
        )
        s3.upload_file(
            data["output"], upload_bucket, os.path.join(upload_path, data["output"])
        )
        s3.upload_file(
            self.logs_path, upload_bucket, os.path.join(upload_path, self.logs_path)
        )


# if __name__ == "__main__":
#     from datetime import datetime
#     import boto3

#     flag_callback = CustomFlagging(flagging_dir="flagged_bonus")
#     s3 = boto3.client("s3")
#     _, _, OUTPUT_BUCKET, OUTPUT_DIR, FLAGGED_DIR = os.environ.get("flagged_dir").split("/")

#     st = datetime.now()
#     image = np.zeros((32,32,3))
#     out = {i: i for i in range(10)}
#     et = datetime.now()

#     inf_time = (et - st).total_seconds() * 10 ** 3
#     flag_callback.flag(
#         im=image,
#         output=out,
#         flag_data={"timestamp": 1234, "infrence_time (ms)": inf_time},
#     )
#     flag_callback.push_to_s3(s3, OUTPUT_BUCKET, f"{OUTPUT_DIR}/{FLAGGED_DIR}")
