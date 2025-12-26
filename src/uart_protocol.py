import struct

class UART():
    def __init__(self, ser, HEADER):
        self.ser = ser
        self.HEADER = HEADER
        self.line_detect_mode = 0
        self.line_error = 0
        self.sign_id = 0

    def send_uart(self, line_detect_mode, line_error, sign_id):
        data_frame = struct.pack("<Bbhb", self.HEADER, line_detect_mode, line_error, sign_id) 
        self.ser.write(data_frame)

