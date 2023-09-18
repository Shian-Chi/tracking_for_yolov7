from myProject.motorInit import s_t
from myProject.pid import PID_Ctrl
from myProject.parameter import Parameters
from myProject.motor import motorCtrl, motorSend, motorRecv
from myProject.distance import Distance
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import datetime
import serial

motor1 = motorCtrl(1)
motor2 = motorCtrl(2)
para = Parameters()
pid = PID_Ctrl()

center_X = para.HD_Width / 2
center_Y = para.HD_Height / 2
t = 0

ser = None


def serialPort():
    global ser
    ser = serial.Serial(
        port='/dev/ttyTHS0',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1 / 115200 * 8 * 12
    )


def dataSend(x,y):
    anglesPerPixel_X = para.FOV / para.HD_Width
    anglesPerPixel_Y = para.FOV / para.HD_Height

    data = f'{anglesPerPixel_X}, {anglesPerPixel_Y}, {x}, {y}'.encode("UTF-8")
    ser.send(data)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=para.HD_Width,
    capture_height=para.HD_Height,
    display_width=para.HD_Width,
    display_height=para.HD_Height,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def motorPID_Ctrl(frameCenter_X, frameCenter_Y):
    pidErr = pid.pid_run(frameCenter_X, frameCenter_Y)
    if abs(pidErr[0]) != 0:
        data1 = motor1.IncrementTurnVal(int(pidErr[0]*100))
        motorSend(data1, 10)
    if abs(pidErr[1]) != 0:
        data2 = motor2.IncrementTurnVal(int(pidErr[1]*100))
        motorSend(data2, 10)
    dataSend()
    print(f"{pidErr[0]:.3f}", f"{pidErr[1]:.3f}")

    if pidErr[0] == pidErr[1] == 0:
        e1 = motor1.getEncoderAndAngle()
        e2 = motor2.getEncoderAndAngle()
        print(e1, e2)
        return e1, e2


def PID(xyxy):
    e = None
    if xyxy is not None:
        # Calculate the center point of the image frame
        e = motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return e


class YOLO():
    def __init__(self, opt):
        self.weights, self.view_img, self.imgsz, self.trace, self.conf_thres, self.iou_thres, self.agnostic, self.augment = \
            opt.weights, opt.view_img, opt.img_size, not opt.no_trace, opt.conf_thres, opt.iou_thres, opt.agnostic_nms, opt.augment

        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.img, self.im0s = [None], None

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)

        if self.half:
            self.model.half()  # to FP16

        # Set Dataloader
#        self.view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        self.t1, self.t2, self.t3 = 0, 0, 0

        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.fourcc = cv2.VideoWriter_fourcc(
            *'mp4v')  # Define the codec for the video
        self.frameOut = None

    def yolo(self):
        # Inference
        self.t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(self.img, augment=self.augment)[0]
        self.t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic)
        self.t3 = time_synchronized()
        return pred

    def runYOLO(self):
        pred = self.yolo()

        for i, det in enumerate(pred):  # detections per image
            s, im0 = '%g: ' % i, self.im0s[i].copy()
            max_conf = -1  # Variable to store the maximum confidence value
            max_xyxy = None  # Variable to store the xyxy with the maximum confidence
            n = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    self.img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    # detections per class
                    n = (det[:, -1] == c).sum()
                    # add to string
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.view_img:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, self.colors[int(
                            cls)], label, line_thickness=1)

                    if conf > max_conf:
                        max_conf = conf
                        max_xyxy = xyxy
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3*(self.t2-self.t1)):.1f}ms) Inference, ({(1E3*(self.t3-self.t2)):.1f}ms) NMS, FPS:{1E3/((1E3*(self.t2-self.t1))+(1E3*(self.t3-self.t2))+s_t):.1f}\n')

        return n, im0, max_conf, max_xyxy

    def loadimg(self):
        self.imgs = [None] * 1
        if not self.cap.isOpened():
            return False
        _, self.imgs[0] = self.cap.read()

        self.im0s = self.imgs.copy()

        # Letterbox
        self.img = [letterbox(x, 640, auto=True, stride=32)[0] for x in self.im0s]

        # Stack
        self.img = np.stack(self.img, 0)

        # Convert
        # BGR to RGB, to bsx3x416x416
        self.img = self.img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        self.img = np.ascontiguousarray(self.img)

        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.half() if self.half else self.img.float()  # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)
        return True

    def save(self, frame):
        t = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        outputPath = f'output_video_{t}.mp4'
        try:
            if self.frameOut is None:
                self.frameOut = cv2.VideoWriter(
                    outputPath, self.fourcc, 30, (para.HD_Width, para.HD_Height))
            self.frameOut.write(frame)
        except Exception as e:
            print("save Error: %s" % e)

    def trackStart(self):
        num = 0
        try:
            if not self.loadimg():
                raise StopIteration
            num, im0, _, xyxy = self.runYOLO()
            PID(xyxy)
#            self.save(im0)
            # Stream results
            '''
            cv2.imshow("media", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise StopIteration
            '''
            return num
        except Exception as e:
            self.cap.release()
            if self.frameOut is not None:
                self.frameOut.release()
            cv2.destroyAllWindows()
            print(f"ERROR Info: {e}\n")
            raise StopIteration


def argument(target: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=target, help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    opt = parser.parse_args()
    # print(opt)
    return opt


def detect_init():
    opt = argument('tennisv7.pt')
    return opt


def get_detect(p):
    return p.trackStart()


if __name__ == '__main__':
    opt = detect_init()
    Tracker = YOLO(opt)
    while True:
        print(get_detect(Tracker))
