#ifndef ENCODER_H
#define ENCODER_H

#include <Arduino.h>
#include "config.h"

class EncoderRPM {
public:
  volatile long count = 0;
  int pinA, pinB;
  int dir;
  float rpm = 0;
  unsigned long lastTime = 0;

  EncoderRPM(int a, int b, int direction)
    : pinA(a), pinB(b), dir(direction) {}

  void begin() {
    pinMode(pinA, INPUT_PULLUP);
    pinMode(pinB, INPUT_PULLUP);
  }

  inline void IRAM_ATTR handleISR() {
    int A = digitalRead(pinA);
    int B = digitalRead(pinB);

    if (A == HIGH) {
      count += (B == LOW) ? 1 : -1;
    } else {
      count += (B == LOW) ? -1 : 1;
    }
  }

  float update() {
    unsigned long now = millis();
    float dt = (now - lastTime) / 1000.0;
    if (dt < ENCODER_DT_MS / 1000.0) return rpm;

    noInterrupts();
    long pulse = count;
    count = 0;
    interrupts();

    rpm = dir * (pulse / PULSE_PER_REV) * 60.0 / dt;
    lastTime = now;
    return rpm;
  }
};

#endif
