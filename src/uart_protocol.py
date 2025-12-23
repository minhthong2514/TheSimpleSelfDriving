import struct

class UART():
    def __init__(self, ser, HEADER):
        self.ser = ser
        self.HEADER = HEADER
        self.line_detect = 0
        self.line_error = 0
        self.sign_id = 0

    def send_uart(self, line_detect, line_error, sign_id):
        data_frame = struct.pack("<BBHB", self.HEADER, line_detect, line_error, sign_id) 
        self.ser.write(data_frame)

