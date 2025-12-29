import struct

class UART():
    def __init__(self, ser, HEADER):
        self.ser = ser
        self.HEADER = HEADER
        self.mode = 0
        self.line_error = 0

    def send_uart(self, mode, line_error):
        data_frame = struct.pack("<Bbh", self.HEADER, mode, line_error) 
        self.ser.write(data_frame)

