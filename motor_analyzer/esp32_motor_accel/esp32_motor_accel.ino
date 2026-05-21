/**
 * MotorSense ESP32 Firmware — v2.0 with On-Device Feature Extraction + OTA
 * Hardware: ESP32 + ADXL345 + Motor Driver + 2× DC Motors
 *
 * Pin Wiring:
 *   Motor 1 → IN1=12, IN2=13, ENA=14
 *   Motor 2 → IN3=25, IN4=26, ENB=27
 *   ADXL345 → SDA=21, SCL=22 (default ESP32 I2C)
 *
 * Serial Protocol:
 *   Raw mode (default): one float per line = Z-axis m/s² @ 100Hz
 *   Feature mode: "F:rms,p2p,variance,crest,zcr,band0,...,band7" @ ~1.5Hz
 *   Status:      "# ..." lines ignored by Python
 *   Commands:    "1"/"2"/"0"/"SPEED:xxx"/"?", "FEAT:1"/"FEAT:0", "OTA"
 *
 * OTA Update:
 *   Send "OTA" over serial → ESP32 switches to OTA listen mode
 *   Upload via Arduino IDE (network port) or ESP32 HTTP OTA
 *
 * Required Libraries:
 *   - Adafruit ADXL345 (by Adafruit)
 *   - Adafruit Unified Sensor (by Adafruit)
 *   - arduinoFFT (by Enrique Condes)
 *   - ArduinoOTA (built-in ESP32)
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <arduinoFFT.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <EEPROM.h>

// ── Pin Assignments ──
const int IN1 = 12, IN2 = 13, ENA = 14;
const int IN3 = 25, IN4 = 26, ENB = 27;

// ── PWM Config ──
const int PWM_FREQ       = 30000;
const int PWM_RESOLUTION = 8;
const int M1_CHANNEL     = 0;
const int M2_CHANNEL     = 1;

// ── Config ──
#define BAUD_RATE        115200
#define SAMPLE_MS        10        // 100 Hz
#define DEFAULT_SPEED    200
#define FEATURE_WINDOW   64        // 64 samples = 0.64s per feature vector
#define FFT_SIZE         64
#define NUM_BANDS        8

// ── WiFi (set via EEPROM or defaults) ──
#define EEPROM_SIZE      256
#define WIFI_SSID_ADDR   0
#define WIFI_PASS_ADDR   64

const char* default_ssid = "MotorSense";
const char* default_pass = "motorsense123";

// ── OTA Config ──
const char* ota_hostname = "motorsense-esp32";
const int ota_port = 3232;

// ── Globals ──
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

uint8_t  motorState  = 0;
uint8_t  cruiseSpeed = DEFAULT_SPEED;
bool     accelReady  = false;
unsigned long lastSample = 0;
bool     featureMode = false;      // false = raw stream, true = feature stream

float    sampleBuffer[FFT_SIZE];
uint8_t  sampleIdx = 0;
uint8_t  featureCount = 0;

arduinoFFT FFT = arduinoFFT();

// ── Motor Functions ──
void killAllMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); ledcWrite(ENA, 0);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW); ledcWrite(ENB, 0);
  motorState = 0;
  Serial.println("# STATUS: ALL STOPPED");
}

void runMotor1Only(uint8_t speed) {
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW); ledcWrite(ENB, 0);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  ledcWrite(ENA, speed);
  motorState = 1;
  Serial.print("# STATUS: MOTOR1 ON speed="); Serial.println(speed);
}

void runMotor2Only(uint8_t speed) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); ledcWrite(ENA, 0);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  ledcWrite(ENB, 255); delay(80);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(ENB, 255); delay(600);
  ledcWrite(ENB, speed);
  motorState = 2;
  Serial.print("# STATUS: MOTOR2 ON speed="); Serial.println(speed);
}

void updateSpeed(uint8_t speed) {
  cruiseSpeed = speed;
  if (motorState == 1) { ledcWrite(ENA, speed); Serial.print("# STATUS: MOTOR1 speed="); Serial.println(speed); }
  else if (motorState == 2) { ledcWrite(ENB, speed); Serial.print("# STATUS: MOTOR2 speed="); Serial.println(speed); }
}

void printStatus() {
  Serial.print("# STATUS: motor="); Serial.print(motorState);
  Serial.print(" speed="); Serial.print(cruiseSpeed);
  Serial.print(" accel="); Serial.print(accelReady ? "OK" : "FAIL");
  Serial.print(" mode="); Serial.println(featureMode ? "FEATURE" : "RAW");
}

// ── On-Device Feature Extraction ──
void computeAndSendFeatures() {
  if (sampleIdx < FFT_SIZE) return;

  double vReal[FFT_SIZE];
  double vImag[FFT_SIZE];
  float sum = 0, sumSq = 0, minVal = 1e10, maxVal = -1e10;
  int zcr = 0;

  for (int i = 0; i < FFT_SIZE; i++) {
    float v = sampleBuffer[i];
    sum += v;
    sumSq += v * v;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
    vReal[i] = v;
    vImag[i] = 0;
  }

  float mean = sum / FFT_SIZE;
  float rms = sqrt(sumSq / FFT_SIZE);
  float p2p = maxVal - minVal;
  float variance = sumSq / FFT_SIZE - mean * mean;
  float crest = (rms > 0.001) ? maxVal / rms : 0;

  for (int i = 1; i < FFT_SIZE; i++) {
    if ((sampleBuffer[i] >= 0 && sampleBuffer[i-1] < 0) ||
        (sampleBuffer[i] < 0 && sampleBuffer[i-1] >= 0)) zcr++;
  }

  FFT.windowing(vReal, FFT_SIZE, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, FFT_SIZE, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, FFT_SIZE);

  float bands[NUM_BANDS];
  int bandSize = (FFT_SIZE / 2) / NUM_BANDS;
  for (int b = 0; b < NUM_BANDS; b++) {
    float e = 0;
    int start = b * bandSize + 1;
    int end = (b == NUM_BANDS - 1) ? FFT_SIZE / 2 : start + bandSize;
    for (int i = start; i < end && i < FFT_SIZE / 2; i++) e += vReal[i];
    bands[b] = e / (end - start + 1);
  }

  Serial.print("F:");
  Serial.print(rms, 4); Serial.print(",");
  Serial.print(p2p, 4); Serial.print(",");
  Serial.print(variance, 4); Serial.print(",");
  Serial.print(crest, 4); Serial.print(",");
  Serial.print(zcr); Serial.print(",");
  for (int b = 0; b < NUM_BANDS; b++) {
    Serial.print(bands[b], 2);
    if (b < NUM_BANDS - 1) Serial.print(",");
  }
  Serial.println();
}

// ── WiFi / OTA Init ──
void initWiFi() {
  String ssid = default_ssid;
  String pass = default_pass;
  EEPROM.begin(EEPROM_SIZE);
  char buf[64];
  for (int i = 0; i < 63; i++) {
    buf[i] = EEPROM.read(WIFI_SSID_ADDR + i);
    if (buf[i] == 0) break;
  }
  buf[63] = 0;
  if (strlen(buf) > 1) ssid = String(buf);
  for (int i = 0; i < 63; i++) {
    buf[i] = EEPROM.read(WIFI_PASS_ADDR + i);
    if (buf[i] == 0) break;
  }
  buf[63] = 0;
  if (strlen(buf) > 1) pass = String(buf);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid.c_str(), pass.c_str());
  Serial.print("# WiFi connecting to "); Serial.print(ssid);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(" OK IP:"); Serial.println(WiFi.localIP());
  } else {
    Serial.println(" FAIL (continuing without WiFi)");
  }
}

void initOTA() {
  ArduinoOTA.setHostname(ota_hostname);
  ArduinoOTA.setPort(ota_port);
  ArduinoOTA.onStart([]() {
    String type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
    Serial.println("# OTA: Start " + type);
  });
  ArduinoOTA.onEnd([]() { Serial.println("# OTA: Done"); });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("# OTA: %u%%\r", progress / (total / 100));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("# OTA: Error %u\n", error);
  });
  ArduinoOTA.begin();
}

// ── Serial Command Parser ──
void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "1" || cmd == "MOTOR1") { runMotor1Only(cruiseSpeed); }
  else if (cmd == "2" || cmd == "MOTOR2") { runMotor2Only(cruiseSpeed); }
  else if (cmd == "0" || cmd == "STOP") { killAllMotors(); }
  else if (cmd == "?" || cmd == "STATUS") { printStatus(); }
  else if (cmd.startsWith("SPEED:") || cmd.startsWith("S:")) {
    int ci = cmd.indexOf(':');
    if (ci >= 0) {
      int v = cmd.substring(ci+1).toInt();
      if (v >= 0 && v <= 255) updateSpeed(v);
      else Serial.println("# ERR: Speed 0-255");
    }
  }
  else if (cmd == "FEAT:1") {
    featureMode = true;
    sampleIdx = 0;
    Serial.println("# MODE: FEATURE (on-device extraction)");
  }
  else if (cmd == "FEAT:0") {
    featureMode = false;
    Serial.println("# MODE: RAW (float stream)");
  }
  else if (cmd == "OTA") {
    Serial.println("# OTA: Starting OTA listen mode...");
    initOTA();
    for (int i = 0; i < 120; i++) {
      ArduinoOTA.handle();
      delay(500);
      if (i % 10 == 0) Serial.print("# OTA: listening...\n");
    }
    Serial.println("# OTA: listen timeout, resuming normal operation");
  }
}

// ── Setup ──
void setup() {
  Serial.begin(BAUD_RATE); delay(300);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  ledcAttachChannel(ENA, PWM_FREQ, PWM_RESOLUTION, M1_CHANNEL);
  ledcAttachChannel(ENB, PWM_FREQ, PWM_RESOLUTION, M2_CHANNEL);
  killAllMotors();

  Wire.begin(); Wire.setClock(400000);

  if (!accel.begin()) {
    Serial.println("# ERR: ADXL345 not found!");
    accelReady = false;
  } else {
    accel.setRange(ADXL345_RANGE_16_G);
    accel.setDataRate(ADXL345_DATARATE_800_HZ);
    accelReady = true;
    Serial.println("# ADXL345 OK @ ±16g 800Hz internal");
  }

  initWiFi();

  Serial.println("# MotorSense v2 Ready");
  Serial.println("# Commands: 1, 2, 0, SPEED:xxx, ?, FEAT:0/1, OTA");
  if (featureMode) Serial.println("# MODE: FEATURE");
  else Serial.println("# MODE: RAW (float stream)");
}

// ── Loop ──
void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  if (WiFi.status() == WL_CONNECTED) {
    ArduinoOTA.handle();
  }

  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_MS) {
    lastSample = now;

    if (accelReady) {
      sensors_event_t event;
      accel.getEvent(&event);
      float z = event.acceleration.z;

      if (featureMode) {
        if (sampleIdx < FFT_SIZE) {
          sampleBuffer[sampleIdx++] = z;
        }
        featureCount++;
        if (featureCount >= FEATURE_WINDOW) {
          computeAndSendFeatures();
          sampleIdx = 0;
          featureCount = 0;
        }
      } else {
        Serial.println(z, 4);
      }
    } else {
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
