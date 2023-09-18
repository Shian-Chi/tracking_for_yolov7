from yolo.visionDetect import YOLO, argument, get_detect
from time import sleep as delay
import sys
sys.path.append('./yolo/')


opt = argument('tennisv7.pt')
Tracker = YOLO(opt)


if __name__ == '__main__':
    while True:
        t = get_detect(Tracker)
#        if t <= 50: # 20FPS
#            delay((50-t)/1E3)
