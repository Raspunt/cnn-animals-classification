import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an animal classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "train", help="Train and select the best epoch using validation data"
    )
    test = commands.add_parser("test", help="Evaluate the saved held-out test split")
    test.add_argument("--model", help="Checkpoint path (default: out/best/trained.pth)")
    predict = commands.add_parser("predict", help="Classify a single image")
    predict.add_argument("image", help="Image path")
    predict.add_argument(
        "--model", help="Checkpoint path (default: out/best/trained.pth)"
    )
    webcam = commands.add_parser(
        "webcam", help="Classify webcam frames (requires the webcam extra)"
    )
    webcam.add_argument(
        "--model", help="Checkpoint path (default: out/best/trained.pth)"
    )
    commands.add_parser("clear-test-data", help="Delete test records from the database")
    commands.add_parser(
        "clear-training-data",
        aliases=["clear-traning-data"],
        help="Delete training records and experiment checkpoints; preserve the best model",
    )
    return parser


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            from PIL import Image

            from nn import transform
            from nn.nn_file_manager import NNFileManager
            from nn.nn_predicer import predict

            model, checkpoint = NNFileManager().load_nn(args.model)
            with Image.open(args.image) as image:
                prediction = predict(
                    transform(image.convert("RGB")), checkpoint["classes"], model
                )
            print(f"Predicted class: {prediction.label}")
            print(f"Model confidence (softmax): {100 * prediction.confidence:.2f}%")
            print("Confidence is not measured accuracy or visual similarity.")
            for name, probability in sorted(
                prediction.probabilities.items(), key=lambda item: -item[1]
            ):
                print(f"  {name}: {100 * probability:.2f}%")
            return
        if args.command == "webcam":
            from camera.webcam import start_webcam

            start_webcam(args.model)
            return
        from commands.cmd_views import CmdViews
        from db import init_db

        init_db()
        views = CmdViews()
        if args.command == "train":
            views.train()
        elif args.command == "test":
            views.test(args.model)
        elif args.command == "clear-test-data":
            views.clear_test_data()
        else:
            views.clear_train_data()
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
