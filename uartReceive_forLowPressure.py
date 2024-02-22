import serial
import serial.tools.list_ports
from threading import Thread
import time
from queue import Queue
from UserProtocolHandle import ProtocolDatasFIFO, collect_raw_packet
from single_line_data_for_real_time import *
import pytz
from datetime import datetime
import cv2


def reshape_pressuremat_values(pressuremat_values):
    new_values = np.zeros((32, 64))
    for i in range(16):
        new_values[:, i * 4:(i + 1) * 4] = pressuremat_values[i]

    return new_values


def LowPressureRealTimeDisplay(pressure_lines):
    sum_arr = np.zeros(16)
    num = 0
    mat_2D = np.zeros(32, 64)

    for pressure_line in pressure_lines:
        pressure_time, pressure_value = process_pressure_values(pressure_line)
        sum_arr += pressure_value
        num += 1
    avg_pressure = sum_arr / num

    mat_2D = reshape_pressuremat_values(avg_pressure)

    output_image = np.reshape(mat_2D, (32, 64))
    normalized_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    color_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
    resized_color_image = cv2.resize(color_image, (640, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('low pressure distribution', resized_color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


pressDataList = []


class userUartReceive:
    def __init__(self):
        self.ser = serial.Serial()
        self.uartReceiveThread = Thread(target=self.uartReceiveTask)

    def port_open(self):
        self.ser.port = 'COM4'
        self.ser.baudrate = 115200
        self.ser.inter_byte_timeout = 0.01
        self.ser.timeout = 2
        try:
            self.ser.open()
        except:
            print("此串口不能被打开")
            return None

    def uartReceiveTask(self):
        global pressDataList
        while True:
            if self.ser.is_open:
                try:
                    data = self.ser.read(10000)
                except:
                    continue

                if len(data):
                    t = time.time()
                    # print("uartReceiveTask rec data", int(round(t * 1000)), data)
                    packet = bytearray()
                    cData = bytearray(data)
                    timestamp = datetime.now().strftime("[%H:%M:%S.%f]")[:-4] + "]"
                    while len(cData):
                        packet.clear()
                        collect_raw_packet(cData, packet)
                        curData = timestamp + packet.hex().upper()
                        if len(packet) == 35:
                            pressDataList.append(curData)
                            # if len(pressDataList) < 10:
                            #     pressDataList.append(curData)
                            # elif len(pressDataList) == 10:
                            #     pressDataList[0:9] = pressDataList[1:-1]
                            #     pressDataList[-1] = curData
                        elif len(packet) == 17:
                            LowPressureRealTimeDisplay(pressDataList)
                            pressDataList = []

            else:
                time.sleep(0.2)
                print("此串口未被打开")


if __name__ == "__main__":
    userUart = userUartReceive()
    userUart.port_open()
    userUart.uartReceiveThread.start()
