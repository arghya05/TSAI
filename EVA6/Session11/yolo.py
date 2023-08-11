import os

import cv2
import numpy as np


class LoadYOLO:
    def __init__(self, labelPath, cfgPath, weightPath, input_w, input_h, threshold=0.5):
        self.labels = open(labelPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(69)
        # self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(cfgPath, weightPath)
        self.output_layers_names = self.net.getUnconnectedOutLayersNames()
        self.threshold = threshold
        self.input_w = input_w
        self.input_h = input_h
        self.colors = np.random.uniform(0, 255, size=(len(self.labels), 3))

    def detect(self, image):
        # self.image = cv2.imread(imPlate)
        self.image = image
        (self.H, self.W) = self.image.shape[:2]

        self.blob = cv2.dnn.blobFromImage(
            self.image, 1 / 255.0, (self.input_w, self.input_h), swapRB=True, crop=False
        )
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        self.net.setInput(self.blob)
        self.layerOutputs = self.net.forward(self.output_layers_names)

        return self.process_output(self.layerOutputs)

    def process_output(self, layerOutputs):
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over layer outputs
        for output in layerOutputs:
            # lopp over each detection
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, 0.3)
        bboxes = []
        outlabels = []
        conf = []
        # ensure at least one detection exists
        if len(indexes) > 0:
            for i in indexes.flatten():
                # extract the bounding box coordinates
                x, y, w, h = boxes[i]
                label = f"{self.labels[classIDs[i]]}"
                confidence = round(confidences[i], 3)

                bboxes.append([x, y, w, h])
                outlabels.append(label)
                conf.append(confidence)

            return bboxes, outlabels, conf
        else:
            return "None Detected"

    def plotbbox(self, boxes, classes, confidence):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            color = self.colors[i]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)

            text = f"{classes[i]}: {confidence[i]}"
            cv2.putText(self.image, text, (x, y - 5), font, 0.5, color, 2)

        return self.image
