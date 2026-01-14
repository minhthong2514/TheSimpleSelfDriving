#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

/* ================= PWM ================= */
#define PWM_RESOLUTION 10
#define PWM_FREQ 100
#define MAX_PWM 1024

/* ================= ENCODER ================= */
#define PULSE_PER_REV 330.0
#define ENCODER_DT_MS 50
#define PID_DT_MS 50

/* ================= MOTOR PINS ================= */
// RL
#define IN1A_RL 25
#define IN2A_RL 33
// RR
#define IN3A_RR 27
#define IN4A_RR 14
// FL
#define IN1B_FL 21
#define IN2B_FL 19
// FR
#define IN3B_FR 18
#define IN4B_FR 5

/* ================= ENCODER PINS ================= */
// A
#define FL_ENC_A 22
#define FR_ENC_A 4
#define RL_ENC_A 32
#define RR_ENC_A 12
// B
#define FL_ENC_B 23
#define FR_ENC_B 15
#define RL_ENC_B 26
#define RR_ENC_B 13

/* ================= ENCODER DIR ================= */
/* PCB: FL ngược chiều */
#define ENC_DIR_FL -1
#define ENC_DIR_FR 1
#define ENC_DIR_RL 1
#define ENC_DIR_RR 1


/* ================= PID GAINS ================= */
#define Kp_FL 22.8
#define Ki_FL 0.6
#define Kd_FL 0.01

#define Kp_FR 21.5
#define Ki_FR 0.7
#define Kd_FR 0.01

#define Kp_RL 22.0
#define Ki_RL 0.8
#define Kd_RL 0.01

#define Kp_RR 22.6
#define Ki_RR 1.0
#define Kd_RR 0.01

/* ================= PID OUTPUT LPF ================= */
#define PID_OUT_ALPHA 0.2  // 0.2 ~ 0.4

#endif
