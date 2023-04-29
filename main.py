import cv2
import numpy as np
from Detector import FaceDetector
from MaskGenerator import MaskGenerator

cap = cv2.VideoCapture(0)
detector = FaceDetector()
maskGenerator = MaskGenerator()

def showImages(actual, target, output1, output2):
    img_actual = np.copy(actual)
    img_target = np.copy(target)
    img_out1 = np.copy(output1)
    img_out2 = np.copy(output2)
    # 640x480 -> 360x480
    img_actual = img_actual[:, 140:500]
    img_out1 = img_out1[:, 140:500]
    # 480x640 -> 360x480
    img_target = cv2.resize(img_target, (360, 480), interpolation=cv2.INTER_AREA)
    img_out2 = cv2.resize(img_out2, (360, 480), interpolation=cv2.INTER_AREA)

    h1 = np.concatenate((img_actual, img_target, img_out1, img_out2), axis=1)

    cv2.imshow('Face Mask', h1)

# Target
#target_image, target_alpha = detector.load_target_img("images/cage.png")
target_image, target_alpha = detector.load_target_img("images/obama.png")
#target_image, target_alpha = detector.load_target_img("images/trump.png")
#target_image, target_alpha = detector.load_target_img("images/kim.png")
#target_image, target_alpha = detector.load_target_img("images/putin.png")
target_landmarks, _, target_face_landmarks= detector.find_face_landmarks(target_image)
target_image_out = detector.drawLandmarks(target_image, target_face_landmarks)

maskGenerator.calculateTargetInfo(target_image, target_alpha, target_landmarks)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks, image, face_landmarks = detector.find_face_landmarks(frame)
    if len(landmarks) == 0:
        continue

    detector.stabilizeVideoStream(frame, landmarks)

    output = maskGenerator.applyTargetMask(frame, landmarks)
    output2 = maskGenerator.applyTargetMaskToTarget(landmarks)

    image_out = detector.drawLandmarks(image, face_landmarks)
    showImages(image_out, target_image_out, output, output2)

    cv2.waitKey(1)