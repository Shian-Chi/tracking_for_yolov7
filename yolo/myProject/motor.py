from yolo.myProject.motorInit import motorSet
from yolo.myProject.parameter import Parameters

import numpy as np
import time
import os
import cv2
import struct
import serial
from time import sleep as delay

para = Parameters()

motor = motorSet()

ser = serial.Serial()

X_Center = 1920/2
Y_Center = 1080/2

rxBuffer = np.zeros(12, dtype="uint8")
HC = np.uint8(62)  # header Code
MOTOR_LEFT = np.uint8(0)
MOTOR_RIGHT = np.uint8(1)


ANGLE_NEW = np.uint16(0)

framePos = np.zeros(4, dtype="uint64")


class motorInformation():
    def __init__(self, ID):
        self.ID = ID
        self.encoder = None
        self.angle = None
        self.speed = None
        self.voltage = None
        self.powers = None
        self.current = None


class motorCtrl():
    def __init__(self, motor_id):
        self.info = motorInformation(motor_id)
        self.ID = np.uint8(motor_id)

    def Stop(self):
        cmd = 129  # 0x81
        data = struct.pack("5B", HC, cmd, self.ID, 0, HC+cmd+self.ID+0)
        motor.send(data, 5)
        return rxBuffer

    def SingleTurnVal(self, dir, value):
        global rxBuffer
        cmd = np.uint8(165)  # 0xA5
        check_sum = Checksum(value+dir)
        value = np.uint16(value)
        buffer = struct.pack("6BH2B", HC, cmd, self.ID, 4,
                             HC + cmd+self.ID+4, dir, value, 0, check_sum)
        return buffer

    def IncrementTurnVal(self, value):
        d = []
        cmd = np.uint8(167)  # 0xA7
        check_sum = Checksum(value)
        buffer = struct.pack("<5BiB", HC, cmd, self.ID, 4,
                             HC+cmd+self.ID+4, value, check_sum)
        '''
        unbuf = struct.unpack("<10B", buffer)
        for i in range (len(unbuf)):
            d.append(hex(int(unbuf[i])))
        print(f'{d}')
        '''
        return buffer

    def motorZero(self, dir):
        data = struct.pack("10B", 62, 165, self.ID, 4,
                           Checksum(62+165+self.ID+4), dir, 0, 0, 0, dir)
        motor.send(data, 10)
        delay(0.1)
                                    
    def getEncoderAndAngle(self):
        cmd = np.uint8(144)  # 0x90
        check_sum = HC+144+self.ID+0
        data = struct.pack("5B", HC, cmd, self.ID, 0, check_sum)
        n = 1
        while n:
            try:
                motor.send(data, 5)
                delay(0.0002)
                r = motor.recv()
                r = struct.unpack(str(len(r))+'B',r)
                if len(r) >= 12:
                    #print("ID:", self.ID, "len", len(r), "data:", r)
                    i = -1
                    if 62 in r:  # 62 is HC
                        i = r.index(HC)
                    if i >= 0:
                        if r[i+2] == self.ID:
                           print("test4")
                           self.info.encoder = r[i+6] << 8 | r[i+5]
                           self.info.angle = self.info.encoder / 91.0
                           #print(f"ID: {self.ID}, Encoder: {self.info.encoder}, Angle: {self.info.angle}")
                           break

                if n == 3:
                    break
                n += 1
            except Exception as e:
                print("ERROR:", e)
                return self.info.angle

            return self.info.angle


def calc_value_Checksum(value):
    value = value & 0xFFFFFFFF
    return value & 0xFF

def Checksum(value):
    val = np.int32(value)
    arr = np.array([val >> 24 & 0xFF, val >> 16 & 0xFF, val >> 8 & 0xFF, val & 0xFF], dtype="uint8")
    total = np.sum(arr)
    check_sum = np.uint8(total & np.uint8(0xFF))
    return np.uint8(check_sum)


def motorSend(data, size):
    return motor.send(data, size)


def motorRecv():
    return motor.recv()


ctrl_1 = motorCtrl(1)
ctrl_2 = motorCtrl(2)
def direction(self,dir):
    d = None
    if dir == 'w' or dir == 'W':
        d = ctrl_2.IncrementTurnVal(-100)
    if dir == 'a' or dir == 'A':
        d = ctrl_1.IncrementTurnVal(100)
    if dir == 's' or dir == 'S':
        d = ctrl_2.IncrementTurnVal(-100)
    if dir == 'd' or dir == 'D':
        d = ctrl_1.IncrementTurnVal(100)        
    if d is not None:
        motor.send(d)