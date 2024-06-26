
import torch
import cv2

import nn
from nn.nn_predicer import predict
from settings import settings



def frame_to_tensor(frame):

    img = Image.fromarray(frame)
    img = transform(img)

    img_tensor = img.unsqueeze(0).to(device)
    return img_tensor

def start_webcam(idx_to_class):
    nn.net.load_state_dict(torch.load(settings.trained_model_path))
    vid = cv2.VideoCapture(0)

    while (True):

        ret, frame = vid.read()
        tensor = nn.frame_to_tensor(frame)
        class_prediction = predict(tensor, idx_to_class)
        print(class_prediction)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()