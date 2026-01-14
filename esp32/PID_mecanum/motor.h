#ifndef MOTOR_H
#define MOTOR_H

#include <Arduino.h>
#include <PID_v1.h>
#include "config.h"

class MotorPID {
public:
  int pinFwd, pinBwd;
  int motorDir;

  double input = 0;
  double output = 0;
  double output_f = 0;
  double setpoint = 0;

  PID pid;

  MotorPID(int fwd, int bwd,
           double kp, double ki, double kd,
           int dir = 1)
    : pinFwd(fwd),
      pinBwd(bwd),
      motorDir(dir),
      pid(&input, &output, &setpoint, kp, ki, kd, DIRECT) {}

  void begin() {
    pinMode(pinFwd, OUTPUT);
    pinMode(pinBwd, OUTPUT);

    ledcAttach(pinFwd, PWM_FREQ, PWM_RESOLUTION);
    ledcAttach(pinBwd, PWM_FREQ, PWM_RESOLUTION);

    pid.SetMode(AUTOMATIC);
    pid.SetSampleTime(PID_DT_MS);
    pid.SetOutputLimits(-MAX_PWM, MAX_PWM);

    stop();
  }

  inline double lpf(double prev, double in) {
    return PID_OUT_ALPHA * in + (1.0 - PID_OUT_ALPHA) * prev;
  }

  void update(double feedbackRPM) {
    input = feedbackRPM;
    pid.Compute();

    output_f = lpf(output_f, output);
    setPWM(motorDir * output_f);
  }

  void setPWM(int pwm) {
    pwm = constrain(pwm, -MAX_PWM, MAX_PWM);

    if (pwm > 0) {
      ledcWrite(pinFwd, pwm);
      ledcWrite(pinBwd, 0);
    } else if (pwm < 0) {
      ledcWrite(pinFwd, 0);
      ledcWrite(pinBwd, -pwm);
    } else {
      stop();
    }
  }

  void stop() {
    ledcWrite(pinFwd, 0);
    ledcWrite(pinBwd, 0);
  }
};

#endif
