#include "config.h"
#include "encoder.h"
#include "motor.h"
#include <PID_v1.h>

/* ================= UART ================= */
#define UART_HEADER 0xAA
#define FRAME_SIZE 4
#define UART_TIMEOUT_MS 300

int16_t uart_line_error = 0;  // error line từ Jetson
unsigned long last_uart_time = 0;

/* ================= LINE PID ================= */
double lineErr = 0;
double lineOut = 0;
double lineSet = 0;

double Kp_line = 0.3;
double Ki_line = 0.0;
double Kd_line = 0.15;

PID pidLine(&lineErr, &lineOut, &lineSet, Kp_line, Ki_line, Kd_line, DIRECT);

/* ================= ENCODERS ================= */
EncoderRPM encFL(FL_ENC_A, FL_ENC_B, ENC_DIR_FL);
EncoderRPM encFR(FR_ENC_A, FR_ENC_B, ENC_DIR_FR);
EncoderRPM encRL(RL_ENC_A, RL_ENC_B, ENC_DIR_RL);
EncoderRPM encRR(RR_ENC_A, RR_ENC_B, ENC_DIR_RR);

/* ================= MOTORS ================= */
MotorPID motorFL(IN1B_FL, IN2B_FL, Kp_FL, Ki_FL, Kd_FL);
MotorPID motorFR(IN3B_FR, IN4B_FR, Kp_FR, Ki_FR, Kd_FR);
MotorPID motorRL(IN1A_RL, IN2A_RL, Kp_RL, Ki_RL, Kd_RL);
MotorPID motorRR(IN3A_RR, IN4A_RR, Kp_RR, Ki_RR, Kd_RR);

/* ================= TARGET ================= */
float base_speed = 50;       // PID bánh
float base_speed_turn = 50;  // tốc độ quay tại chỗ
float delta_speed = 0;       // PID line

float targetFL = 0;
float targetFR = 0;
float targetRL = 0;
float targetRR = 0;


/* ================= STATE ================= */
bool prev_mode_line = false;

enum RobotMode : uint8_t {
  MODE_STOP = 1,
  MODE_LINE = 0,
  MODE_TURN_LEFT = 3,
  MODE_TURN_RIGHT = 4,
  MODE_TURN_AROUND = 2
};

RobotMode robotMode = MODE_STOP;


/* ================= ISR ================= */
void IRAM_ATTR ISR_FL() {
  encFL.handleISR();
}
void IRAM_ATTR ISR_FR() {
  encFR.handleISR();
}
void IRAM_ATTR ISR_RL() {
  encRL.handleISR();
}
void IRAM_ATTR ISR_RR() {
  encRR.handleISR();
}

/* ================= UART PARSER ================= */
void readUART() {
  static uint8_t buf[FRAME_SIZE];
  static uint8_t idx = 0;
  static bool syncing = false;

  while (Serial.available()) {
    uint8_t c = Serial.read();

    if (!syncing) {
      if (c == UART_HEADER) {
        buf[0] = c;
        idx = 1;
        syncing = true;
      }
    } else {
      buf[idx++] = c;

      if (idx >= FRAME_SIZE) {
        robotMode = (RobotMode)buf[1];
        uart_line_error = (int16_t)(buf[2] | (buf[3] << 8));

        last_uart_time = millis();
        syncing = false;
      }
    }
  }
}

/* ================= SETUP ================= */
void setup() {
  Serial.begin(115200);

  encFL.begin();
  encFR.begin();
  encRL.begin();
  encRR.begin();
  attachInterrupt(digitalPinToInterrupt(FL_ENC_A), ISR_FL, RISING);
  attachInterrupt(digitalPinToInterrupt(FR_ENC_A), ISR_FR, RISING);
  attachInterrupt(digitalPinToInterrupt(RL_ENC_A), ISR_RL, RISING);
  attachInterrupt(digitalPinToInterrupt(RR_ENC_A), ISR_RR, RISING);

  motorFL.begin();
  motorFR.begin();
  motorRL.begin();
  motorRR.begin();

  pidLine.SetMode(AUTOMATIC);
  pidLine.SetSampleTime(PID_DT_MS);
  pidLine.SetOutputLimits(-30, 30);

  Serial.println("MECANUM + LINE PID + UART READY");
}

/* ================= LOOP ================= */
void loop() {
  readUART();

  /* ================= FAIL SAFE ================= */
  if (millis() - last_uart_time > UART_TIMEOUT_MS) {
    robotMode = MODE_STOP;
  }

  /* ================= UPDATE RPM ================= */
  float rpmFL = encFL.update();
  float rpmFR = encFR.update();
  float rpmRL = encRL.update();
  float rpmRR = encRR.update();

  /* =====================================================
     MODE LINE – BÁM LINE (ƯU TIÊN)
     ===================================================== */
  if (robotMode == MODE_LINE) {

    if (!prev_mode_line) {
      pidLine.SetMode(MANUAL);
      lineOut = 0;
      pidLine.SetMode(AUTOMATIC);
    }

    lineErr = uart_line_error;
    pidLine.Compute();
    delta_speed = lineOut;

    targetFL = base_speed - delta_speed;
    targetFR = base_speed + delta_speed;
    targetRL = base_speed - delta_speed;
    targetRR = base_speed + delta_speed;

    motorFL.setpoint = targetFL;
    motorFR.setpoint = targetFR;
    motorRL.setpoint = targetRL;
    motorRR.setpoint = targetRR;

    motorFL.update(rpmFL);
    motorFR.update(rpmFR);
    motorRL.update(rpmRL);
    motorRR.update(rpmRR);

    prev_mode_line = true;
  }

  /* =====================================================
     MODE TURN – QUAY TẠI CHỖ
     ===================================================== */
  else if (
    robotMode == MODE_TURN_LEFT || robotMode == MODE_TURN_RIGHT || robotMode == MODE_TURN_AROUND) {

    pidLine.SetMode(MANUAL);
    prev_mode_line = false;

    float turn_speed = base_speed_turn;

    if (robotMode == MODE_TURN_LEFT || robotMode == MODE_TURN_AROUND) {
      turn_speed = -turn_speed;
    }

    targetFL = turn_speed;
    targetFR = -turn_speed;
    targetRL = turn_speed;
    targetRR = -turn_speed;

    motorFL.setpoint = targetFL;
    motorFR.setpoint = targetFR;
    motorRL.setpoint = targetRL;
    motorRR.setpoint = targetRR;

    motorFL.update(rpmFL);
    motorFR.update(rpmFR);
    motorRL.update(rpmRL);
    motorRR.update(rpmRR);
  }

  /* =====================================================
     MODE STOP – ĐỨNG YÊN
     ===================================================== */
  else {  // MODE_STOP
    pidLine.SetMode(MANUAL);
    prev_mode_line = false;

    motorFL.stop();
    motorFR.stop();
    motorRL.stop();
    motorRR.stop();
  }

  /* ================= DEBUG ================= */
  Serial.printf(
    "MODE %d | ERR %d | DELTA %.1f\n",
    robotMode, uart_line_error, delta_speed);

  delay(PID_DT_MS);
}
