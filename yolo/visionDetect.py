from .myProject.motorInit import s_t
from .myProject.pid import PID_Ctrl
from .myProject.parameter import Parameters
from .myProject.motor import motorCtrl, motorSend
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .models.experimental import attempt_load
from .utils.datasets import letterbox
from .utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, time_synchronized, TracedModel
import numpy as np
import datetime

motor1 = motorCtrl(1)
motor2 = motorCtrl(2)
para = Parameters()
pid = PID_Ctrl()

center_X = para.HD_Width / 2
center_Y = para.HD_Height / 2
t = 0


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=para.HD_Width,
    capture_height=para.HD_Height,
    display_width=para.HD_Width,
    display_height=para.HD_Height,
    framerate=60,
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
    print(f"{pidErr[0]:.3f}", f"{pidErr[1]:.3f}")

    if pidErr[0] == pidErr[1] == 0:
        _, angle1 = motor1.getEncoderAndAngle()
        _, angle2 = motor2.getEncoderAndAngle()
        print(angle1, angle2)
        return angle1, angle2
    return 0, 0


def PID(xyxy):
    angle1, angle2 = 0, 0
    if xyxy is not None:
        # Calculate the center point of the image frame
        angle1, angle2 = motorPID_Ctrl(((xyxy[0] + xyxy[2]) / 2).item(), ((xyxy[1] + xyxy[3]) / 2).item())
    return angle1, angle2


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
        self.view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        self.t1, self.t2, self.t3 = 0, 0, 0
        self.spendTime = 0
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
            detectFlag = False
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
                        plot_one_box(xyxy, im0, self.colors[int(cls)], label, line_thickness=1)

                    if conf > max_conf:
                        detectFlag = True
                        max_conf = conf
                        max_xyxy = xyxy
            # Print time (inference + NMS)
            inferenceTime = 1E3*(self.t2-self.t1)
            NMS_Time = 1E3*(self.t3-self.t2)
            self.spendTime = inferenceTime + NMS_Time + s_t
            fps = 1E3/self.spendTime
            print(f'{s}Done. ({inferenceTime:.1f}ms) Inference, ({NMS_Time:.1f}ms) NMS, FPS:{fps:.1f}\n')

        return fps, detectFlag, max_xyxy

    def loadimg(self):
        imgs = [None] * 1
        ret, imgs[0] = self.cap.read()
        
        if not ret:
            grabbed = self.cap.grab()
            if grabbed:
                ret, imgs[0] = self.cap.retrieve()
                if not ret:
                    return False 
        self.im0s = imgs.copy()

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

    def trackingStart(self):
        num = 0
        try:
            if not self.loadimg():
                print("No image")
                raise StopIteration
            fps, flag, xyxy = self.runYOLO()
            angle1, angle2 = PID(xyxy)

            # Save image
#            self.save(im0)

            # Stream results
            if self.view_img:
                #cv2.imshow("media", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise StopIteration
                
            return fps, flag, angle1, angle2
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
    return p.trackingStart()


'''
if __name__ == '__main__':
    opt = detect_init()
    Tracker = YOLO(opt)
    while True:
        print(get_detect(Tracker))
'''

