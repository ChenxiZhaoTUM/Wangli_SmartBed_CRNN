import serial

ser = serial.Serial('COM6', 115200)  # 创建一个串口对象
while True:
    if ser.in_waiting > 0:  # 如果串口有数据可读取
        # data = ser.readline().decode().strip()  # 读取串口数据并转成字符串形式
        # print(data)  # 打印读取到的数据

        data_hex = ser.read(ser.in_waiting)  # 读取所有可用数据
        print("Hex Data:", data_hex.hex())  # 打印十六进制数据

        # decimal_data = int.from_bytes(data_hex, byteorder='big')  # 将字节数据转换成十进制
        # print("Decimal Data:", decimal_data)  # 打印十进制数据
