"""
Perform single object tracking.
After detecting the object, outliers are removed based on previous location.
"""

from argparse import ArgumentParser
from time import time
import numpy as np
import cv2
from tqdm import tqdm

import torch

from torchvision import transforms

from model import centernet, draw_fps


if __name__ == "__main__":
    parser = ArgumentParser(description="Track any sport ball")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for ball detection")
    args = parser.parse_args()

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 8

    DELTA = 2  # tracking const variable

    assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Define and load model
    model = centernet()
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    PRED_CENTER = None

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    prev_time = time()

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process image to feed the model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # quick
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))  # quick
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

        with torch.no_grad():
            hm, offset = model(input_tensor.to(device).float().unsqueeze(0))

        hm = torch.sigmoid(hm)
        hm = hm.cpu().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().numpy().squeeze(0)

        # Basic tracking ideas:
        # if max(score) is very high, we take it as the position of the ball
        # if max(score) is very low, we assume no ball is on the image
        # otherwise we take the max(score) in a vicinity of the previous known position of the ball
        if np.max(hm, axis=None) < 0.01:
            PRED_CENTER = None
        elif np.max(hm, axis=None) > 0.4:
            PRED_CENTER = np.unravel_index(np.argmax(hm, axis=None), hm.shape)
        else:
            PRED_CENTER = None
            if last_loc is not None:
                sub_hm = hm[
                    last_loc[0] - DELTA : last_loc[0] + DELTA,
                    last_loc[1] - DELTA : last_loc[1] + DELTA,
                ]
                if np.max(sub_hm, axis=None) > 0.02:
                    PRED_CENTER = np.unravel_index(
                        np.argmax(sub_hm, axis=None), sub_hm.shape
                    ) + (np.array(last_loc) - DELTA)
                    PRED_CENTER = tuple(PRED_CENTER)

        if PRED_CENTER is not None:
            last_loc = PRED_CENTER
            score = hm[PRED_CENTER]

            arr = (
                np.array(
                    [PRED_CENTER[1], PRED_CENTER[0]]
                    + offset[:, PRED_CENTER[0], PRED_CENTER[1]]
                )
                * MODEL_SCALE
            )

            # for point, score in zip(points, scores):
            u = round(arr[0] * frame.shape[1] / INPUT_WIDTH)
            v = round(arr[1] * frame.shape[0] / INPUT_HEIGHT)

            cv2.circle(frame, (u, v), radius=10, color=(255, 0, 0), thickness=3)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (u + 15, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 0, 0),
                thickness=3,
            )

        fps = 1 / (time() - prev_time)
        prev_time = time()
        draw_fps(frame, fps)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"video/image_{idx:03d}.png", frame)
