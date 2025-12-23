import time
from uart_protocol import UART
import serial


HEADER = 0xAA

ser = serial.Serial(port="/dev/ttyUSB0",baudrate=115200, timeout=1)
uart = UART(ser, HEADER)
# ====== GIÁ TRỊ CỐ ĐỊNH ======
line_detect = 1          # int8
line_error  = 1234       # uint16
sign_id     = 3          # int8

while True:
    line_detect = 1          # int8
    line_error  = 1234       # uint16
    sign_id     = 3          # int8

    uart.send_uart(line_detect, line_error, sign_id)
    time.sleep(1)   # 10Hz

    line_detect = 0          # int8
    line_error  = 1234       # uint16
    sign_id     = 3          # int8

    uart.send_uart(line_detect, line_error, sign_id)
    time.sleep(1)

