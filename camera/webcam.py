from PIL import Image

from nn import transform
from nn.nn_file_manager import NNFileManager
from nn.nn_predicer import predict


def frame_to_tensor(frame):
    import cv2

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)


def start_webcam(model_path=None):
    try:
        import cv2
    except ImportError as error:
        raise ValueError("Webcam support requires: uv sync --extra webcam") from error
    model, checkpoint = NNFileManager().load_nn(model_path)
    video = cv2.VideoCapture(0)
    try:
        if not video.isOpened():
            raise ValueError("Cannot open webcam 0.")
        while True:
            success, frame = video.read()
            if not success:
                break
            prediction = predict(frame_to_tensor(frame), checkpoint["classes"], model)
            print(f"{prediction.label}: confidence {100 * prediction.confidence:.2f}%")
            cv2.imshow("Animal classifier (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
