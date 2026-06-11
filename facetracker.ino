/*
 * FaceTrak — Pan-Tilt Servo Controller
 * Protokoll: "P090T045\n"  (Pan=90°, Tilt=45°)
 * Antwort:   "OK 90 45\n" oder "ERR\n"
 */

#include <Servo.h>

// ──── Konfiguration ──────────────────────────────────────────
const uint8_t PAN_PIN  = 9;
const uint8_t TILT_PIN = 10;

const bool PAN_REVERSE  = false;
const bool TILT_REVERSE = true;

const int PAN_MIN = 0,  PAN_MAX = 180;
const int TILT_MIN = 30, TILT_MAX = 150;

// Grad pro Update-Schritt (alle UPDATE_MS). Höher = schneller, ruckeliger.
const int      MAX_STEP  = 3;
const uint16_t UPDATE_MS = 15;

const long BAUD = 115200;

// ──── Intern ─────────────────────────────────────────────────
Servo panServo, tiltServo;

int panTarget = 90, tiltTarget = 90;   // Soll (nach Reverse/Clamp)
int panActual = 90, tiltActual = 90;   // Ist (was am Servo anliegt)

char buf[16];
uint8_t bufLen = 0;
unsigned long lastUpdate = 0;

int clampReverse(int v, int lo, int hi, bool rev) {
  v = constrain(v, lo, hi);
  return rev ? (hi - (v - lo)) : v;
}

bool parseCommand(const char* s) {
  // Erwartet: P<int>T<int>
  if (s[0] != 'P') return false;
  char* end;
  long p = strtol(s + 1, &end, 10);
  if (end == s + 1 || *end != 'T') return false;
  const char* tStart = end + 1;
  long t = strtol(tStart, &end, 10);
  if (end == tStart || *end != '\0') return false;

  panTarget  = clampReverse((int)p, PAN_MIN,  PAN_MAX,  PAN_REVERSE);
  tiltTarget = clampReverse((int)t, TILT_MIN, TILT_MAX, TILT_REVERSE);
  return true;
}

int stepToward(int actual, int target) {
  int diff = target - actual;
  if (abs(diff) <= MAX_STEP) return target;
  return actual + (diff > 0 ? MAX_STEP : -MAX_STEP);
}

void setup() {
  Serial.begin(BAUD);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panActual);
  tiltServo.write(tiltActual);
  Serial.println(F("READY"));
}

void loop() {
  // Serielle Daten non-blocking einsammeln
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufLen > 0) {
        buf[bufLen] = '\0';
        if (parseCommand(buf)) {
          Serial.print(F("OK "));
          Serial.print(panTarget);
          Serial.print(' ');
          Serial.println(tiltTarget);
        } else {
          Serial.println(F("ERR"));
        }
        bufLen = 0;
      }
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = c;
    } else {
      bufLen = 0;  // Overflow → Müll verwerfen
    }
  }

  // Servos sanft Richtung Ziel bewegen
  unsigned long now = millis();
  if (now - lastUpdate >= UPDATE_MS) {
    lastUpdate = now;
    int newPan  = stepToward(panActual, panTarget);
    int newTilt = stepToward(tiltActual, tiltTarget);
    if (newPan != panActual)   { panActual = newPan;   panServo.write(panActual); }
    if (newTilt != tiltActual) { tiltActual = newTilt; tiltServo.write(tiltActual); }
  }
}
