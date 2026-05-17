/**
 * ═══════════════════════════════════════════════════════════
 *  MotorSense ESP32 Firmware — Web App Compatible
 *  Hardware: ESP32 + ADXL345 + Motor Driver + 2× DC Motors
 * ═══════════════════════════════════════════════════════════
 *
 *  ── YOUR Actual Pin Wiring ────────────────────────────────
 *   Motor 1 → IN1=12, IN2=13, ENA=14
 *   Motor 2 → IN3=25, IN4=26, ENB=27
 *   ADXL345 → SDA=21, SCL=22 (default ESP32 I2C)
 *             VCC=3.3V, GND=GND
 *
 *  ── Serial Commands (from Python web app) ─────────────────
 *   "1"        → Run Motor 1 (stops Motor 2)
 *   "2"        → Run Motor 2 (stops Motor 1, runs anti-stiction)
 *   "0"        → Stop all motors
 *   "SPEED:xxx"→ Set cruise speed 0–255 (e.g. "SPEED:180")
 *   "?"        → Print current status
 *
 *  ── Serial Output ─────────────────────────────────────────
 *   Lines starting with '#' = status/debug (ignored by Python)
 *   Plain float lines       = Z-axis accel in m/s² (100 Hz)
 *
 *  ── Required Libraries (Arduino Library Manager) ──────────
 *   • Adafruit ADXL345
 *   • Adafruit Unified Sensor
 * ═══════════════════════════════════════════════════════════
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

// ─────────────────────────────────────────────────────────────
//  YOUR Pin Assignments (matched to your working code)
// ─────────────────────────────────────────────────────────────
const int IN1 = 12;
const int IN2 = 13;
const int ENA = 14;

const int IN3 = 25;
const int IN4 = 26;
const int ENB = 27;

// ─────────────────────────────────────────────────────────────
//  PWM Config (matched to your working code)
// ─────────────────────────────────────────────────────────────
const int PWM_FREQ       = 30000;   // 30 kHz
const int PWM_RESOLUTION = 8;       // 8-bit → 0–255
const int M1_CHANNEL     = 0;
const int M2_CHANNEL     = 1;

// ─────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────
#define BAUD_RATE        115200
#define SAMPLE_MS        10       // 100 Hz output to Python
#define DEFAULT_SPEED    200      // Default cruise PWM

// ─────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

uint8_t  motorState  = 0;           // 0=off, 1=motor1, 2=motor2
uint8_t  cruiseSpeed = DEFAULT_SPEED;
bool     accelReady  = false;
unsigned long lastSample = 0;

// ─────────────────────────────────────────────────────────────
//  Motor Functions
// ─────────────────────────────────────────────────────────────

void killAllMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); ledcWrite(ENA, 0);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW); ledcWrite(ENB, 0);
  motorState = 0;
  Serial.println("# STATUS: ALL STOPPED");
}

void runMotor1Only(uint8_t speed) {
  // Kill Motor 2 first
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW); ledcWrite(ENB, 0);

  // Motor 1 forward at requested speed
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  ledcWrite(ENA, speed);

  motorState = 1;
  Serial.print("# STATUS: MOTOR1 ON speed=");
  Serial.println(speed);
}

void runMotor2Only(uint8_t speed) {
  // Kill Motor 1 first
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); ledcWrite(ENA, 0);

  // ── Anti-stiction rocking pulse (your original logic) ──
  // Snap backward 80ms to clear mechanical binding
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENB, 255);
  delay(80);

  // Slam forward at full power 600ms to build rotational inertia
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENB, 255);
  delay(600);

  // Drop to cruise speed
  ledcWrite(ENB, speed);

  motorState = 2;
  Serial.print("# STATUS: MOTOR2 ON speed=");
  Serial.println(speed);
}

void updateSpeed(uint8_t speed) {
  cruiseSpeed = speed;
  if (motorState == 1) {
    ledcWrite(ENA, speed);
    Serial.print("# STATUS: MOTOR1 speed updated=");
    Serial.println(speed);
  } else if (motorState == 2) {
    ledcWrite(ENB, speed);
    Serial.print("# STATUS: MOTOR2 speed updated=");
    Serial.println(speed);
  }
}

void printStatus() {
  Serial.print("# STATUS: motor=");
  Serial.print(motorState);
  Serial.print(" speed=");
  Serial.print(cruiseSpeed);
  Serial.print(" accel=");
  Serial.println(accelReady ? "OK" : "FAIL");
}

// ─────────────────────────────────────────────────────────────
//  Serial Command Parser
// ─────────────────────────────────────────────────────────────
void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "1" || cmd == "MOTOR1") {
    runMotor1Only(cruiseSpeed);
  }
  else if (cmd == "2" || cmd == "MOTOR2") {
    runMotor2Only(cruiseSpeed);
  }
  else if (cmd == "0" || cmd == "STOP") {
    killAllMotors();
  }
  else if (cmd == "?" || cmd == "STATUS") {
    printStatus();
  }
  else if (cmd.startsWith("SPEED:") || cmd.startsWith("S:")) {
    int colonIdx = cmd.indexOf(':');
    if (colonIdx >= 0) {
      int val = cmd.substring(colonIdx + 1).toInt();
      if (val >= 0 && val <= 255) {
        updateSpeed((uint8_t)val);
      } else {
        Serial.println("# ERR: Speed must be 0-255");
      }
    }
  }
  // Unknown commands silently ignored (keeps float stream clean)
}

// ─────────────────────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  delay(300);

  // ── Motor driver GPIO ──
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // ── PWM channels (ESP32 Arduino core v3.x API) ──
  ledcAttachChannel(ENA, PWM_FREQ, PWM_RESOLUTION, M1_CHANNEL);
  ledcAttachChannel(ENB, PWM_FREQ, PWM_RESOLUTION, M2_CHANNEL);

  // Safe start — all motors off
  killAllMotors();

  // ── I2C at 400 kHz Fast Mode ──
  Wire.begin();
  Wire.setClock(400000);

  // ── ADXL345 init ──
  if (!accel.begin()) {
    Serial.println("# ERR: ADXL345 not found! Check SDA(21)/SCL(22) wiring.");
    accelReady = false;
  } else {
    accel.setRange(ADXL345_RANGE_16_G);         // ±16g for high-vibration
    accel.setDataRate(ADXL345_DATARATE_800_HZ); // Internal 800 Hz sampling
    accelReady = true;
    Serial.println("# ADXL345 OK — Range: ±16g @ 800Hz internal");
  }

  Serial.println("# MotorSense Ready. Commands: 1, 2, 0, SPEED:xxx, ?");
  Serial.println("# Output: one float per line (Z-axis m/s^2 @ 100Hz)");
}

// ─────────────────────────────────────────────────────────────
//  Loop
// ─────────────────────────────────────────────────────────────
void loop() {

  // ── Non-blocking serial command reader ──
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  // ── Sample accelerometer at 100 Hz (every 10ms) ──
  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_MS) {
    lastSample = now;

    if (accelReady) {
      sensors_event_t event;
      accel.getEvent(&event);

      // Send Z-axis as plain float — Python reads this stream
      Serial.println(event.acceleration.z, 4);

    } else {
      // Retry ADXL init every 2 seconds
      static unsigned long lastRetry = 0;
      if (now - lastRetry > 2000) {
        lastRetry = now;
        if (accel.begin()) {
          accel.setRange(ADXL345_RANGE_16_G);
          accel.setDataRate(ADXL345_DATARATE_800_HZ);
          accelReady = true;
          Serial.println("# ADXL345 reconnected!");
        }
      }
    }
  }
}
