"""
Perform multi object detection on a mp4 video with a pytorch neural network model (.pth)
"""

from argparse import ArgumentParser
from time import time
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torchvision import transforms

from model import centernet, draw_fps

CONF = 0.5


def on_trackbar(val):
    """Slider to fine-tune your confidence threshold"""
    global CONF
    CONF = val / 100


def select(hm, threshold):
    """
    Keep only local maxima (kind of NMS).
    We make sure to have no adjacent detection in the heatmap.
    """

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 1:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    return pred_centers


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object detection")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for ball detection")
    parser.add_argument(
        "--conf", type=float, default=0.8, help="Threshold to keep an object"
    )
    args = parser.parse_args()

    CONF = args.conf

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 8

    assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Define and load model
    model = centernet()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    TITLE_WINDOW = "Detection"
    cv2.namedWindow(TITLE_WINDOW)
    cv2.createTrackbar(
        "Confidence",
        TITLE_WINDOW,
        int(CONF * 100),
        100,
        on_trackbar,
    )

    prev_time = time()

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process for model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        input_tensor = preprocess(img)

        # Inference
        hm, offset = model(input_tensor.to(device).float().unsqueeze(0))

        hm = torch.sigmoid(hm)
        hm = hm.cpu().detach().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().detach().numpy().squeeze(0)

        # Select detection with confidence threshold
        pred_centers = select(hm, CONF)

        for center in pred_centers:
            arr = (
                np.array([center[1], center[0]] + offset[:, center[0], center[1]])
                * MODEL_SCALE
            )

            u = round(arr[0] * frame.shape[1] / INPUT_WIDTH)
            v = round(arr[1] * frame.shape[0] / INPUT_HEIGHT)

            cv2.circle(frame, (u, v), radius=10, color=(0, 255, 255), thickness=2)

        fps = 1 / (time() - prev_time)
        prev_time = time()
        draw_fps(frame, fps)

        cv2.imshow(TITLE_WINDOW, frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"video/image_{idx:03d}.png", frame)
