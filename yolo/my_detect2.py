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
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
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

class Yolo():
    def __init__(self,args):
        self.args = args
        print("loading YOLOv7 ...")
        # Initialize
        set_logging()
        self.device = select_device(self.args.YOLOdevice)
        self.half = self.device.type != 'cpu'
        # Load model
        self.model = experimental.attempt_load(self.args.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.args.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Second-stage classifier
        if self.args.classify:
            self.modelc = load_classifier(name='resnet101', n=1000)  # initialize
            self.modelc.load_state_dict(torch.load(self.args.weights_classify, map_location='cpu'))
            self.modelc.to(self.device).eval()
        # if webcam:
        _ = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.cap = self.init_video_capture()

    def yolo(self,img):
        im0 = img
        if(len(np.shape(img)) == 3):
            img = np.transpose(img,(2,0,1))
        elif(len(np.shape(img)) == 4):
            img = np.transpose(img,(0,3,1,2))
            if self.args.classify:
                im0 = np.transpose(im0,(0,2,3,1))
                im0.squeeze()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        # if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
        #     # self.old_img_b = img.shape[0]
        #     # self.old_img_h = img.shape[2]
        #     # self.old_img_w = img.shape[3]
        #     self.model(img, augment=self.args.augment)[0]

        # Inference
        # t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.args.augment)[0]
        # t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(
            pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes, agnostic=self.args.agnostic_nms)
        t3 = time_synchronized()
        # print(f'yolo Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        # Apply Classifier
        if self.args.classify:
            print('yolo pred',pred)
            pred = apply_classifier(pred, self.modelc, img, im0)
            print('res pred',pred)
        return pred

    def init_video_capture(self,):
        cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=para.media_width,capture_height=para.media_height,display_width=para.media_width,display_height=para.media_height,framerate=para.FPS,flip_method=0), cv2.CAP_GSTREAMER)
        
        #cap.set(cv2.CAP_PROP_FPS, int(self.args.fps))
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.cap_W)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.cap_H)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"fps:{fps} ,WIDTH:{w},HEIGHT:{h}")
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        return cap
    

    def main(self,):
        
        success, frame = self.cap.read()
        if success is not True:
            print("Read frame failed.")
            exit(-1)



        bbox = None

        while(True):
            success, frame = self.cap.read()
            if(success is not True): 
                print("video done.")
                self.cap.release()
                cv2.destroyAllWindows()
                exit(-1)
            #frame = cv2.resize(frame,(640,480))
        

            # '''YOLO Searching'''

            bbox = self.yolo(frame)
            for object in bbox[0]:        
                print(object)

            cv2.imshow("video",frame)
            if cv2.waitKey(1) == ord("q"):
                self.cap.release()
                cv2.destroyAllWindows()
                raise StopIteration


 


def get_Argument():
    parser = ArgumentParser()
    parser.add_argument('--input',type=str,default=4,help='input VideoCapture')
    parser.add_argument('--cap-H',type=int,default=480,help='VideoCapture size HEIGHT')
    parser.add_argument('--cap-W',type=int,default=640,help='VideoCapture size WIDTH')
    parser.add_argument('--fps', type=int,default=60,help='FPS of the output video')

    '''YOLOv7'''
    parser.add_argument('--weights', nargs='+', type=str,default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640,help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--YOLOdevice', default='',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int,help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',help='augmented inference')
    parser.add_argument('--classify', type=bool,default=False, help='Second-stage classifier')
    parser.add_argument('--weights-classify', nargs='+', type=str,default='resnet101.pth', help='model.pt path(s)')
  

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_Argument()
    Tracker = Yolo(args)
    Tracker.main()
 