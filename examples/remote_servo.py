import serial
import time
import requests

ser = serial.Serial('COM10', 115200, timeout=1)

while True:
    ser.write('H'.encode('utf-8'))
    time.sleep(0.1)