import argparse
import logging
import time
from collections import deque
from multiprocessing import Manager, Process, Value
from typing import Optional, Tuple

import onnxruntime as ort

ort.set_default_logger_severity(4)
import cv2
import numpy as np
from omegaconf import OmegaConf

from constants import classes

logger = logging.getLogger(__name__)


class BaseRecognition:
    def __init__(self, model_path: str, tensors_list, prediction_list):
        self.started = None
        self.output_names = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.model_path = model_path
        self.window_size = None
        self.tensors_list = tensors_list
        self.prediction_list = prediction_list

    def clear_tensors(self):
        """
        Clear the list of tensors.
        """
        for _ in range(self.window_size):
            self.tensors_list.pop(0)

    def run(self):
        """
        Run the recognition model.
        """
        if self.session is None:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[3]
            self.output_names = [output.name for output in self.session.get_outputs()]

        if len(self.tensors_list) >= self.input_shape[3]:
            input_tensor = np.stack(self.tensors_list[: self.window_size], axis=1)[None][None]
            st = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor.astype(np.float32)})[0]
            gloss = str(classes[outputs.argmax()])
            logger.info(f"- Prediction time {round(time.time() - st, 3)}, new gloss: {gloss}")
            if gloss != self.prediction_list[-1] and len(self.prediction_list):
                self.prediction_list.append(gloss)
            self.clear_tensors()
            logging.info(f" --- {len(self.tensors_list)} frames in queue")


class Recognition(BaseRecognition):
    def __init__(self, model_path: str, tensors_list: list, prediction_list: list):
        """
        Initialize recognition model.

        Parameters
        ----------
        model_path : str
            Path to the model.
        tensors_list : List
            List of tensors to be used for prediction.
        prediction_list : List
            List of predictions.

        Notes
        -----
        The recognition model is run in a separate process.
        """
        super().__init__(model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list)
        self.started = True

    def start(self):
        self.run()


class RecognitionMP(Process, BaseRecognition):
    def __init__(self, model_path: str, tensors_list, prediction_list):
        """
        Initialize recognition model.

        Parameters
        ----------
        model_path : str
            Path to the model.
        tensors_list : Manager.list
            List of tensors to be used for prediction.
        prediction_list : Manager.list
            List of predictions.

        Notes
        -----
        The recognition model is run in a separate process.
        """
        super().__init__()
        BaseRecognition.__init__(
            self, model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list
        )
        self.started = Value("i", False)

    def run(self):
        while True:
            BaseRecognition.run(self)


class Runner:
    def __init__(self, model_path: str, config: OmegaConf = None, mp: bool = False) -> None:
        """
        Initialize runner.

        Parameters
        ----------
        model_path : str
            Path to the model.
        config : OmegaConf
            Configuration file.

        Notes
        -----
        The runner uses multiprocessing to run the recognition model in a separate process.

        """
        self.multiprocess = mp
        self.cap = cv2.VideoCapture(0)
        self.manager = Manager() if self.multiprocess else None
        self.tensors_list = self.manager.list() if self.multiprocess else []
        self.prediction_list = self.manager.list() if self.multiprocess else []
        self.prediction_list.append("---")
        self.frame_counter = 0
        self.frame_interval = config.frame_interval
        self.prediction_classes = deque(maxlen=4)
        self.mean = config.mean
        self.std = config.std
        if self.multiprocess:
            self.recognizer = RecognitionMP(model_path, self.tensors_list, self.prediction_list)
        else:
            self.recognizer = Recognition(model_path, self.tensors_list, self.prediction_list)

    def add_frame(self, image):
        """
        Add frame to queue.

        Parameters
        ----------
        image : np.ndarray
            Frame to be added.
        """
        self.frame_counter += 1
        if self.frame_counter == self.frame_interval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize(image, (224, 224))
            image = (image - self.mean) / self.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0

    @staticmethod
    def resize(im, new_shape=(224, 224)):
        """
        Resize and pad image while preserving aspect ratio.

        Parameters
        ----------
        im : np.ndarray
            Image to be resized.
        new_shape : Tuple[int]
            Size of the new image.

        Returns
        -------
        np.ndarray
            Resized image.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        return im

    def run(self):
        """
        Run the runner.

        Notes
        -----
        The runner will run until the user presses 'q'.
        """
        if self.multiprocess:
            self.recognizer.start()

        while self.cap.isOpened():
            if self.recognizer.started:
                _, frame = self.cap.read()
                text_div = np.zeros((50, frame.shape[1], 3), dtype=np.uint8)
                self.add_frame(frame)

                if not self.multiprocess:
                    self.recognizer.start()

                if self.prediction_list:
                    self.prediction_classes.extend(self.prediction_list)
                    text = "  ".join(self.prediction_classes)
                    cv2.putText(text_div, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                if len(self.prediction_list) > 100:
                    self.prediction_list.clear()

                frame = np.concatenate((frame, text_div), axis=0)
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.recognizer.kill()
                    break


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")

    parser.add_argument("-p", "--config", required=True, type=str, help="Path to config")
    parser.add_argument("--mp", required=False, action="store_true", help="Enable multiprocessing")
    parser.add_argument("--log", required=False, action="store_true", help="Enable logging")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    if args.log:
        logging.getLogger().setLevel(logging.INFO)
    conf = OmegaConf.load(args.config)
    runner = Runner(conf.model_path, conf, args.mp)
    runner.run()
