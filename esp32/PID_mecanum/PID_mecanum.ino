#include "config.h"
#include "encoder.h"
#include "motor.h"
#include <PID_v1.h>

/* ================= UART ================= */
#define UART_HEADER 0xAA

int8_t line_enable = 0;
int16_t uart_line_error = 0;
int8_t sign_id = -1;

/* ================= LINE PID ================= */
double lineErr = 0;  // pixel error
double lineOut = 0;  // delta_speed (rpm)
double lineSet = 0;  // luôn = 0

double Kp_line = 0.03;
double Ki_line = 0.0;
double Kd_line = 0.15;

PID pidLine(&lineErr, &lineOut, &lineSet,
            Kp_line, Ki_line, Kd_line, DIRECT);

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
float base_speed = 20;  // rpm
float delta_speed = 0;

float targetFL = 0;
float targetFR = 0;
float targetRL = 0;
float targetRR = 0;

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
  static uint8_t state = 0;
  static uint8_t buf[5];
  static uint8_t idx = 0;

  while (Serial.available()) {
    uint8_t c = Serial.read();

    if (state == 0) {
      if (c == UART_HEADER) {
        buf[0] = c;
        idx = 1;
        state = 1;
      }
    } else {
      buf[idx++] = c;
      if (idx >= 5) {
        line_enable = (int8_t)buf[1];
        uart_line_error = (int16_t)(buf[2] | (buf[3] << 8));
        sign_id = (int8_t)buf[4];
        state = 0;
      }
    }
  }
}

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
  pidLine.SetOutputLimits(-30, 30);  // giới hạn delta rpm
  pidLine.SetSampleTime(PID_DT_MS);

  Serial.println("MECANUM + LINE PID + UART READY");
}

void loop() {
  readUART();

  /* ===== UPDATE RPM ===== */
  float rpmFL = encFL.update();
  float rpmFR = encFR.update();
  float rpmRL = encRL.update();
  float rpmRR = encRR.update();

  /* ===== LINE PID ===== */
  if (line_enable) {
    lineErr = (double)uart_line_error;
    pidLine.Compute();
    delta_speed = lineOut;
  } else {
    delta_speed = 0;
  }

  /* ===== MIX SPEED ===== */
  targetFL = -base_speed + delta_speed;
  targetRL = -base_speed + delta_speed;
  targetFR = base_speed - delta_speed;
  targetRR = base_speed - delta_speed;

  /* ===== MOTOR PID ===== */
  motorFL.setpoint = targetFL;
  motorFR.setpoint = targetFR;
  motorRL.setpoint = targetRL;
  motorRR.setpoint = targetRR;

  motorFL.update(rpmFL);
  motorFR.update(rpmFR);
  motorRL.update(rpmRL);
  motorRR.update(rpmRR);

  /* ===== DEBUG ===== */
  Serial.printf(
    "EN %d ERR %d DELTA %.1f | FL %.1f FR %.1f RL %.1f RR %.1f\n",
    line_enable, uart_line_error, delta_speed,
    rpmFL, rpmFR, rpmRL, rpmRR);

  delay(PID_DT_MS);
}
