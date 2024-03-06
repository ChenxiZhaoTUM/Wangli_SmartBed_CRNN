import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import serial
import serial.tools.list_ports
import UserProtocolHandle as matUser
from single_line_data_for_real_time import process_pressure_values
import DeviceUserProtocolHandle as deviceUser
from DeviceUserExample import packetHandleThread
import time
from queue import Queue
from threading import Thread, Lock, Event
from datetime import datetime

uartFifo = Queue(200)
cmdAckEvent = Event()
uartPacketFifo = deviceUser.ProtocolDatasFIFO()
matFifo = matUser.ProtocolDatasFIFO()
pressDataList = []


class SmartBedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, control_port='COM3', control_baudrate=115200,
                 pressure_port='COM4', pressure_baudrate=115200):
        self.alpha = 0.1
        self.episode = 1
        self.output_episode = 100
        self.time_per_action = 0.1

        self.action_space = spaces.MultiDiscrete([4] * 6)  # 0:no change, 1:inflation, 2:stop, 3:deflation
        low_obs = np.full(16, 0).astype(np.float32)
        high_obs = np.full(16, 4096).astype(np.float32)  # lack of normalization
        self.observation_space = spaces.Box(low_obs, high_obs)
        self.obs = np.zeros(16)
        self.previous_pressure_values = np.zeros(16)
        # self.previous_action = np.zeros(6)  # here need to change to the previous inner pressure of the airbag
        self.pressDataList = []

        # Serial port initialization for communication with the smart bed hardware
        self.running = True
        self.obsLock = Lock()
        self.cmdLock = Lock()
        self.get_uart_ports()

        self.pressure_ser = serial.Serial()
        self.pressure_ser.port = pressure_port
        self.pressure_ser.baudrate = pressure_baudrate
        self.pressure_ser.inter_byte_timeout = 0.01
        self.pressure_ser.timeout = 2
        self.open_serial_port(self.pressure_ser, "pressure")

        self.matThread = Thread(target=self.mat_task)
        self.matThread.start()

        self.control_ser = serial.Serial()
        self.control_ser.port = control_port
        self.control_ser.baudrate = control_baudrate
        self.control_ser.inter_byte_timeout = 0.01
        self.control_ser.timeout = 2
        self.open_serial_port(self.control_ser, "control")

        self.uartReceiveThread = Thread(target=self.uart_receive_task)
        self.uartReceiveThread.start()
        self.packetParaseThread = packetHandleThread()
        self.packetParaseThread.airPressReflash.connect(self.airbag_pressure_display)
        self.packetParaseThread.start()

    def get_uart_ports(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("Error finding any port!")
            return None
        for port in ports:
            print(f"Finding {port.description}")

    def open_serial_port(self, ser, port_type):
        try:
            ser.open()
            print(f"{port_type.capitalize()} serial port opened successfully.")
        except Exception as e:
            print(f"Error opening {port_type} serial port: {e}")

    def close_serial_port(self, ser, port_type):
        try:
            ser.close()
            print(f"{port_type.capitalize()} serial port closed successfully.")
        except Exception as e:
            print(f"Error closing {port_type} serial port: {e}")

    def stop_threads(self):
        self.running = False
        self.uartReceiveThread.join()
        self.packetParaseThread.join()

    def mat_task(self):
        while self.running:
            # here delete while True
            if self.pressure_ser.is_open:
                try:
                    data = self.pressure_ser.read(1000)
                    if len(data):
                        # t = time.time()
                        # print("uartReceiveTask rec data", int(round(t * 1000)), data)
                        packet = bytearray()
                        cData = bytearray(data)
                        timestamp = datetime.now().strftime("[%H:%M:%S.%f]")[:-4] + "]"
                        while len(cData):
                            packet.clear()
                            matUser.collect_raw_packet(cData, packet)
                            curData = timestamp + packet.hex().upper()
                            if len(packet) == 35:
                                self.pressDataList.append(curData)
                            elif len(packet) == 17:
                                sum_arr = np.zeros(16)
                                for pressure_line in self.pressDataList:
                                    pressure_time, pressure_value = process_pressure_values(pressure_line)
                                    sum_arr += pressure_value

                                if self.pressDataList is not None:
                                    with self.obsLock:
                                        self.obs = sum_arr / len(self.pressDataList)
                                self.pressDataList = []
                except Exception as e:
                    print(f"Error reading pressure data: {e}")

        else:
            time.sleep(0.2)
            print("Cannot open mat port!")

    def uart_receive_task(self):
        while self.running:
            if self.control_ser.is_open:
                try:
                    data = self.control_ser.read(10000)
                except:
                    continue
                t = time.time()
                # print("UartReceiveTask rec data", int(round(t * 1000)), data)
                if len(data):
                    uartFifo.put(data)
                    uartPacketFifo.enqueue(data)
            else:
                time.sleep(0.1)

    def cmd_packet_exec(self, cmdPacket):
        self.cmdLock.acquire()
        if self.control_ser.is_open:
            print("control command ser.write:", cmdPacket.hex())
            try:
                self.control_ser.write(cmdPacket)
            except:
                print("Control command ser.write fail!")
                # QMessageBox.critical(self, "Cmd Error", "write fail")
                self.cmdLock.release()
                return False
            if cmdAckEvent.wait(1.0):
                cmdAckEvent.clear()
                self.cmdLock.release()
                print("cmd ack ok")
                return True
            else:
                print("cmd ack timeout")
        self.cmdLock.release()

    def airbag_pressure_display(self, data):
        for i in range(6):
            print("index[%d]: %.2f" % (i, data[i]))

    def send_airbag_control_command(self, action):
        print("Action is: ", action)
        if not isinstance(action, (list, np.ndarray)):
            action = [action]

        index = 0
        cfgTime = 0XFF  # 1-20(S) or 0XFF(always run)
        mapByte = deviceUser.airMap()
        mapByte.char = 0  # resets all airbag controls to 0

        # Update mapByte based on the action for each airbag
        airbag_mapping = ['xiaoTui', 'daTui', 'Tun', 'Yao', 'Xiong', 'Jian']  # according to map_bits
        for i, act in enumerate(action):
            if act > 0:
                setattr(mapByte.bit, airbag_mapping[i], 1)  # eg. mapByte.bit.xiaoTui = 1

        if mapByte.char == 0:
            print("No airbag control action specified.")
            return

        # action_code: 1: inflation, 2: stop, 3: deflation
        if 1 in action:
            action_code = 1
        elif 2 in action:
            action_code = 2
        elif 3 in action:
            action_code = 3
        else:
            action_code = 0

        print("action_code: ", action_code)
        print("mapByte.char: ", mapByte.char)

        # Convert the action to the corresponding command packet
        cmdPacketData = deviceUser.airControlCmdPacketSend(index, action_code, mapByte.char, cfgTime)
        if not self.cmd_packet_exec(cmdPacketData):
            print("Cmd Error: No response!")
        else:
            print("Cmd success!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_reward = 0.0
        self.action_time = 0.0
        self.action_time_steps = 0
        self.obs = np.zeros(16)  # pressure
        self._get_obs = self.obs.astype(np.float32)
        print('reset train_obs: ', self._get_obs)
        return self._get_obs, {}

    def step(self, action):
        with self.obsLock:
            pressure_values = self.obs.copy()

            reward = 0.0
            done = False
            self.action_time_steps += 1
            self.action_time += self.time_per_action

            # action: 0:no change, 1:inflation, 2:stop, 3:deflation
            # take action of 6 airbags

            self.send_airbag_control_command(action)

            if pressure_values is None:
                print("Failed to read pressure data, skipping step.")
                return self._get_obs, reward, done, False, {}

            self.obs = pressure_values  # Update pressure in observation
            pressure_variance = np.var(pressure_values)
            pressure_change_continuity = np.mean(np.abs(self.previous_pressure_values - pressure_values))
            self.previous_pressure_values = pressure_values.copy()

            # action_change_continuity = np.mean(np.abs(self.previous_action - action))
            # self.previous_action = action.copy()

            # set reward
            # pressure distribution
            if pressure_variance == 0:
                reward += 10.0
            else:
                reward += 1.0 / (pressure_variance + 1)  # prevent division by 0

            # reward -= (pressure_change_continuity + action_change_continuity)
            reward -= pressure_change_continuity

            self.total_reward += reward

            print("self.obs:", self.obs)
            if np.all(self.obs == 0):
                done = True
                self.episode += 1

            return self._get_obs, reward, done, False, {}

    def __del__(self):
        self.stop_threads()
        self.close_serial_port(self.pressure_ser, "pressure")
        self.close_serial_port(self.control_ser, "control")


register(
    id='SmartBedEnv-v0',
    entry_point='smart_bed_env:SmartBedEnv',
)
