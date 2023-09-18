import struct
from myProject.distance import Distance
from myProject.motor import motorCtrl, motorSend, motorRecv
from myProject.pid import PID_Ctrl
from myProject.motorInit import s_t
import argparse
import time
from pathlib import Path
from argparse import ArgumentParser
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from models import experimental
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from myProject.parameter import Parameters
para = Parameters()

motor1 = motorCtrl(1)
motor2 = motorCtrl(2)

para = Parameters()

pid = PID_Ctrl()


center_X = para.HD_Width / 2
center_Y = para.HD_Height / 2
t = 0

dist = Distance(110)


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
        "nvarguscamerasrc sensor-id=%d ! "
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


def trackTime(t1, t2, t3):
    global t
    if int(pid.output[0]) == int(pid.output[1]) == int(0):
        print("Track Time:", t)
        t = 0
    else:
        t += ((1E3 * (t2 - t1)) + (1E3 * (t3 - t2)))
    with open("track time", 'a') as f_t:
        f_t.write(str(t)+'\n')
    return t


def motorPID_Ctrl(frameCenter_X, frameCenter_Y):
    pidErr = pid.pid_run(frameCenter_X, frameCenter_Y)
    if abs(pidErr[0]) != 0:
        data1 = motor1.IncrementTurnVal(int(pidErr[0]*100))
        motorSend(data1, 10)
    if abs(pidErr[1]) != 0:
        data2 = motor2.IncrementTurnVal(int(pidErr[1]*100))
        motorSend(data2, 10)

    print(f"{pidErr[0]:.3f}", f"{pidErr[1]:.3f}")
    '''
    if pidErr[0] == pidErr[1] == 0:
        e1 = motor1.getEncoderAndAngle()
        e2 = motor2.getEncoderAndAngle()
    '''


def PID(xyxy):
    if xyxy is not None:
        # Calculate the center point of the image frame
        motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(),
                      ((xyxy[1] + xyxy[3]) / 2).item())


class Yolo():
    def __init__(self, args):
        self.args = args
        print("loading YOLOv7 ...")
        # Initialize
        set_logging()
        self.device = select_device(self.args.YOLOdevice)
        self.half = self.device.type != 'cpu'
        # Load model
        self.model = experimental.attempt_load(
            self.args.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.args.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Second-stage classifier
        if self.args.classify:
            self.modelc = load_classifier(
                name='resnet101', n=1000)  # initialize
            self.modelc.load_state_dict(torch.load(
                self.args.weights_classify, map_location='cpu'))
            self.modelc.to(self.device).eval()
        # if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once

        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.callBackFlags = False
        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
        
    def callBack(self, flags=False):
        self.callBackFlags = flags
        return self.callBackFlags

    def yolo(self, img):
        if (len(np.shape(img)) == 3):
            img = np.transpose(img, (2, 0, 1))
        elif (len(np.shape(img)) == 4):
            img = np.transpose(img, (0, 3, 1, 2))

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        self.t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.args.augment)[0]
        self.t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,
                                   classes=self.args.classes, agnostic=self.args.agnostic_nms)
        self.t3 = time_synchronized()
        return pred

    def close(self):
        self.cap.release()
        #cv2.destroyAllWindows()

    def detect(self):
        try:
            while True:
                bbox = None
                f = self.callBack(True)
                names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                
                if f:
                    det = None
                    success, frame = self.cap.read()
                    if (success is not True):
                        print("Error: Video Done.")
                        self.close()
                        exit(-1)
                    frame = cv2.resize(frame, (640, 640))

                    '''YOLO Searching'''
                    bbox = self.yolo(frame)
                    print(bbox)
                    max_conf = -1  # Variable to store the maximum confidence value
                    max_xyxy = None  # Variable to store the xyxy with the maximum confidence
                    for i, det in enumerate(bbox):  # detections per image
                        s = '%g: ' % i
                        if len(det):
                            # Print results
                            for c in det[:, -1].unique():
                                # detections per class
                                n = (det[:, -1] == c).sum()
                                # add to string
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if conf > max_conf:
                                    max_conf = conf
                                    max_xyxy = xyxy
                    PID(max_xyxy)
                    print(f'{s}Done. ({(1E3*(self.t2-self.t1)):.1f}ms) Inference, ({(1E3*(self.t3-self.t2)):.1f}ms) NMS, FPS:{1E3/((1E3*(self.t2-self.t1))+(1E3*(self.t3-self.t2))):.1f}\n')
                for *xyxy, _, _ in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)  
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("video", frame)
                if cv2.waitKey(1) == ord("q"):
                    self.close()
                    raise StopIteration

        except:
            self.close()



                                                                                   
def get_Argument():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default=4, help='input VideoCapture')
    parser.add_argument('--cap-H', type=int, default=480, help='VideoCapture size HEIGHT')
    parser.add_argument('--cap-W', type=int, default=640, help='VideoCapture size WIDTH')
    parser.add_argument('--fps', type=int, default=60, help='FPS of the output video')

    '''YOLOv7'''
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--YOLOdevice', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classify', type=bool, default=False, help='Second-stage classifier')
    parser.add_argument('--weights-classify', nargs='+', type=str, default='resnet101.pth', help='model.pt path(s)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_Argument()
    Tracker = Yolo(args)
    Tracker.detect()
