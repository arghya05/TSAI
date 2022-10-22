import unittest

import requests
import json
import base64
from requests import Response


class TestFargateGradio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("deployment.json") as f:
            cls.base_url = json.load(f)["base_url"]

        print(f"using base_url={cls.base_url}\n\n")

        cls.image_paths = ["cat.png", "ship.png", "automobile.png", "dog.png"]

        # convert image to base64

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            with open(image_path, "rb") as f:
                ext = image_path.split('.')[-1]
                prefix = f'data:image/{ext};base64,'
                base64_data = prefix + base64.b64encode(f.read()).decode('utf-8')

            payload = json.dumps({
            "data": [
                base64_data
            ]
            })

            headers = {
            'Content-Type': 'application/json'
            }

            response: Response = requests.request("POST", self.base_url, headers=headers, data=payload, timeout=15)

            print(f"response: {response.text}")

            data = response.json()['data'][0]

            predicted_label = data['label']

            print(f"predicted label: {predicted_label}")

            self.assertEqual(image_path.split(".")[0], predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()