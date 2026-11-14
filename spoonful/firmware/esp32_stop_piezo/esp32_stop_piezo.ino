// Spoonful ESP32-S3: arcade STOP button + piezo throat sensor -> newline JSON over USB serial @115200.
//   {"btn":1} once per press, {"btn":0} on release (25 ms debounce)
//   {"piezo":<envelope>} at 20 Hz : 50 ms RMS of the AC-coupled piezo, sampled at 1 kHz, 0..~2000
//   {"sw":1|0} STOP switch contacts closed|open, on every change (25 ms debounce)
//   {"hb":1,"sw":1|0} every 1 s (the switch state rides along, so a missed change is corrected within 1 s)
// Wiring: arcade button between BTN_PIN and GND; STOP switch (rocker or push button) between STOP_PIN and GND.
// Both use the internal pull-up, so an open circuit (switch off, or a broken wire) reads as "sw":0.
// piezo -> 1 Mohm to GND in parallel, signal into PIEZO_PIN through 10 kohm, biased to mid-rail by 2 x 100 kohm.
// Arduino IDE: board "ESP32S3 Dev Module", USB CDC On Boot: Enabled.

#include <Arduino.h>

const int BTN_PIN = 4;
const int STOP_PIN = 5;           // rocker: closed (I) = run, open (O) = STOP. Push button: closed while pressed
const int PIEZO_PIN = 1;          // ADC1_CH0 on the S3
const uint32_t SAMPLE_US = 1000;  // 1 kHz
const int WINDOW = 50;            // 50 ms RMS window
const uint32_t REPORT_MS = 50;    // 20 Hz
const uint32_t HB_MS = 1000;
const uint32_t DEBOUNCE_MS = 25;

int32_t buf[WINDOW];
int bufIdx = 0;
float dcMean = 2048.0f;           // running mean removes the bias
uint32_t lastSample = 0, lastReport = 0, lastHb = 0, lastBtnChange = 0;
bool btnStable = false, btnRaw = false;
bool swStable = false, swRaw = false;
uint32_t lastSwChange = 0;

void setup() {
  Serial.begin(115200);
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(STOP_PIN, INPUT_PULLUP);
  swStable = swRaw = digitalRead(STOP_PIN) == LOW;
  analogReadResolution(12);
  for (int i = 0; i < WINDOW; i++) buf[i] = 0;
}

void loop() {
  uint32_t nowUs = micros();
  uint32_t nowMs = millis();

  if (nowUs - lastSample >= SAMPLE_US) {
    lastSample += SAMPLE_US;
    int v = analogRead(PIEZO_PIN);
    dcMean += 0.001f * (v - dcMean);
    buf[bufIdx] = v - (int32_t)dcMean;
    bufIdx = (bufIdx + 1) % WINDOW;
  }

  bool pressed = digitalRead(BTN_PIN) == LOW;
  if (pressed != btnRaw) { btnRaw = pressed; lastBtnChange = nowMs; }
  if (btnRaw != btnStable && nowMs - lastBtnChange >= DEBOUNCE_MS) {
    btnStable = btnRaw;
    Serial.println(btnStable ? "{\"btn\":1}" : "{\"btn\":0}");
  }

  bool closed = digitalRead(STOP_PIN) == LOW;
  if (closed != swRaw) { swRaw = closed; lastSwChange = nowMs; }
  if (swRaw != swStable && nowMs - lastSwChange >= DEBOUNCE_MS) {
    swStable = swRaw;
    Serial.println(swStable ? "{\"sw\":1}" : "{\"sw\":0}");
  }

  if (nowMs - lastReport >= REPORT_MS) {
    lastReport = nowMs;
    float acc = 0;
    for (int i = 0; i < WINDOW; i++) acc += (float)buf[i] * buf[i];
    int env = (int)sqrtf(acc / WINDOW);
    Serial.printf("{\"piezo\":%d}\n", env);
  }

  if (nowMs - lastHb >= HB_MS) {
    lastHb = nowMs;
    Serial.println(swStable ? "{\"hb\":1,\"sw\":1}" : "{\"hb\":1,\"sw\":0}");
  }
}
