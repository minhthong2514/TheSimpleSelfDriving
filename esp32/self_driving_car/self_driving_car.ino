#include <Arduino.h>


// Chân điều khiển động cơ
#define IN1A 25
#define IN2A 33
#define IN3A 27
#define IN4A 14

#define IN1B 21
#define IN2B 19
#define IN3B 18
#define IN4B 5

const int RESOLUTION = 12;
const int FREQ = 1000;

#define HEADER 0xAA
#define FRAME_SIZE 5
#define LED_PIN 2

// ===== GIÁ TRỊ DEBUG MONG ĐỢI =====
#define EXP_LINE_ENABLE 0
#define EXP_LINE_ERROR 150
#define EXP_SIGN_ID -1

uint8_t frame[FRAME_SIZE];

int8_t line_enable;
uint16_t line_error;
int8_t sign_id;

void setup() {
  Serial.begin(115200);

  pinMode(13, OUTPUT);
  pinMode(IN1A, OUTPUT);
  pinMode(IN2A, OUTPUT);
  pinMode(IN3A, OUTPUT);
  pinMode(IN4A, OUTPUT);

  pinMode(IN1B, OUTPUT);
  pinMode(IN2B, OUTPUT);
  pinMode(IN3B, OUTPUT);
  pinMode(IN4B, OUTPUT);

  digitalWrite(IN1A, 0);
  digitalWrite(IN2A, 0);
  digitalWrite(IN3A, 0);
  digitalWrite(IN4A, 0);
  digitalWrite(IN1B, 0);
  digitalWrite(IN2B, 0);
  digitalWrite(IN3B, 0);
  digitalWrite(IN4B, 0);

  ledcAttach(IN1A, FREQ, RESOLUTION);
  ledcAttach(IN2A, FREQ, RESOLUTION);
  ledcAttach(IN3A, FREQ, RESOLUTION);
  ledcAttach(IN4A, FREQ, RESOLUTION);

  ledcAttach(IN1B, FREQ, RESOLUTION);
  ledcAttach(IN2B, FREQ, RESOLUTION);
  ledcAttach(IN3B, FREQ, RESOLUTION);
  ledcAttach(IN4B, FREQ, RESOLUTION);

  delay(2000);
}

void loop() {

  if (Serial.available() >= FRAME_SIZE) {

    if (Serial.read() != HEADER) return;

    Serial.readBytes(&frame[1], FRAME_SIZE - 1);

    line_enable = (int8_t)frame[1];
    line_error = frame[2] | (frame[3] << 8);
    sign_id = (int8_t)frame[4];

    // ===== SO SÁNH =====
    if (line_enable == EXP_LINE_ENABLE && line_error <= EXP_LINE_ERROR && sign_id == EXP_SIGN_ID) {
      digitalWrite(13, HIGH);
    } else {
      digitalWrite(13, LOW);
    }
  }
}
