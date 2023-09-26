/*
This program is written to work with the acquisition.py and live_class_2.py programs on the pc.
This program reads 8 analog channel (distributed between ADC1 and ADC2), and sends the resulting 
values with a buffer to the computer.
It's also mean to work with a Adafruit PCA9685 16-Channel Servo Driver to control
servors to make the prostetic hand move. 
I hope the I2C bus wont disrupt anything.
Update: tried with an lcd screen with I2C module. No problem with acquisition.
All the files can be found on GitHub: https://github.com/SartoratoGiulio/ProHand_Programs

Written by Giulio Sartorato.
*/

/*
ToDo: Try to add more functionalities with the button to maybe start calibration or what not. (Example: double click, long press, etc...)
*/

#include <driver/adc.h>


// I only have this now to test I2C.
// It doesn't seem to affect the program
// Change it with a servo shield for live test
#include <LiquidCrystal_I2C.h>
int lcdColumns = 16;
int lcdRows = 2;
LiquidCrystal_I2C lcd(0x3F, lcdColumns, lcdRows);


// BLT serial of HC-05 module
#define ONBOARD_LED 2
HardwareSerial &btSerial = Serial2;  // Rx: 16, Tx: 17. Swap on the bt module

// Reset Button Pin and variables
#define butPin 12  // ADC2_5 //12

// Encoder pins and variables
#define sensPin1 15  // ADC2_3 //15
#define sensPin2 5
#define N 72
#define STEP 360 / N

// EMG pins and variables
#define emg0 14  // ADC2_CHANNEL_6
#define emg1 27  // ADC2_CHANNEL_7
#define emg2 26  // ADC2_CHANNEL_9
#define emg3 25  // ADC2_CHANNEL_8

#define emg4 33  // ADC1_CHANNEL_5
#define emg5 32  // ADC1_CHANNEL_4
#define emg6 35  // ADC1_CHANNEL_7
#define emg7 34  // ADC1_CHANNEL_6
#define emg8 39  // ADC1_CHANNEL_3
#define emg9 36  // ADC1_CHANNEL_0
static uint16_t emgRead0 = 0;
static uint16_t emgRead1 = 0;
static uint16_t emgRead2 = 0;
static uint16_t emgRead3 = 0;
static uint16_t emgRead4 = 0;
static uint16_t emgRead5 = 0;
static uint16_t emgRead6 = 0;
static uint16_t emgRead7 = 0;
static uint16_t emgRead8 = 0;
static uint16_t emgRead9 = 0;

int count = 0;    // Encoder Count
char buffer[45];  // Buffer to send the message

// Calibration Variables
int minAngle = -1;
int maxAngle = -1;
bool calibration = false;

// Variable for timer test
int t = 0;
int t_old = 0;

// Test variable
bool test = false;
bool btEn = false;
bool srEn = false;

// Timer stuff
hw_timer_t *Timer0_Cfg = NULL;
hw_timer_t *Timer1_Cfg = NULL;
bool sample = false;
#define FS 1000
#define APB_CLK 80000000
#define PRESCALER 80
int ticks = APB_CLK / (FS * PRESCALER);


// Interrupt functions
void encoderFunc() {
  if (digitalRead(sensPin1) == digitalRead(sensPin2)) {
    count--;
  } else {
    count++;
  }
}

void IRAM_ATTR timerFunc() {
  sample = true;
}
void testBlink() {
  if (test || calibration) {
    digitalWrite(ONBOARD_LED, !digitalRead(ONBOARD_LED));
  }
}

// Various Functions
void multiAnalogRead() {
  emgRead0 = analogRead(emg0);
  emgRead1 = analogRead(emg1);
  emgRead2 = analogRead(emg2);
  emgRead3 = analogRead(emg3);
  emgRead4 = analogRead(emg4);
  emgRead5 = analogRead(emg5);
  emgRead6 = analogRead(emg6);
  emgRead7 = analogRead(emg7);
}

void resetAnalog() {
  emgRead0 = 0;
  emgRead1 = 0;
  emgRead2 = 0;
  emgRead3 = 0;
  emgRead4 = 0;
  emgRead5 = 0;
  emgRead6 = 0;
  emgRead7 = 0;
  emgRead8 = 0;
  emgRead9 = 0;
}

String poses[] = { "medium wrap", "lateral", "extension type", "tripod", "power sphere", "power disk", "prismatic pinch", "index extension", "thumb adduction", "prismatic four fingers", "wave in", "wave out", "fist", "open hand" };

void poseProcessing(int poseNum) {
  /*
    Handle poses with Adafruit PCA9685
  */
  if (poseNum <= 10) {
    lcd.clear();
    lcd.setCursor(0, 0);  //(column, row)
    lcd.print(poses[poseNum - 1]);
  }
}

void setup() {
  // Can't go with lower baudrate, the buffer print takes too longo otherways :c
  Serial.begin(460800);
  //btSerial.begin(230400); //used when tring thing out with bluetooth. Not fast enough. Keeping it just in case

  lcd.init();
  lcd.backlight();

  Timer0_Cfg = timerBegin(0, PRESCALER, true);  // timer (0 to 3), prescaler, count_up
  timerAttachInterrupt(Timer0_Cfg, &timerFunc, true);
  timerAlarmWrite(Timer0_Cfg, ticks, true);  // which timer, cout at with alarm goes off, autoreload
  timerAlarmEnable(Timer0_Cfg);              // enable alarmed timer

  // Timer to blink onboard led to signal test mode
  Timer1_Cfg = timerBegin(1, PRESCALER, true);
  timerAttachInterrupt(Timer1_Cfg, &testBlink, true);
  timerAlarmWrite(Timer1_Cfg, 500000, true);
  timerAlarmEnable(Timer1_Cfg);

  analogSetAttenuation(ADC_11db);
  analogSetWidth(12);

  pinMode(ONBOARD_LED, OUTPUT);
  pinMode(sensPin1, INPUT_PULLUP);
  pinMode(sensPin2, INPUT_PULLUP);
  pinMode(butPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(sensPin1), &encoderFunc, CHANGE);
  delay(2000);
  digitalWrite(ONBOARD_LED, !test && srEn);
}

void loop() {

  // Reset count if button pressed
  if (!digitalRead(butPin) && !calibration) {
    count = 0;
  }

  if (test) { t = micros(); }
  if (sample && srEn && !calibration) {
    if (!test) {
      sample = false;
      multiAnalogRead();
      // Reduced numer of characters to a minimum to speed up comunication
      sprintf(buffer, "%04d%04d%04d%04d%04d%04d%04d%04d%04d\n", emgRead0, emgRead1, emgRead2, emgRead3, emgRead4, emgRead5, emgRead6, emgRead7, count * STEP);
      Serial.print(buffer);

      //Serial.println(emgRead0);
    } else {
      sample = false;
      sprintf(buffer, "Sampling period: % 5d us\tDeg: %d\n", t - t_old, count * STEP);
      Serial.print(buffer);  //Test if sample frequency is right. Period in us
      t_old = t;
    }
  }

  if (calibration) {
    if (minAngle == -1 && maxAngle == -1 && !digitalRead(butPin)) {
      maxAngle = count * STEP;
      Serial.println(maxAngle);
      delay(1000);
    } else if (minAngle == -1 && maxAngle != -1 && !digitalRead(butPin)) {
      minAngle = count * STEP;
      Serial.println(minAngle);
      delay(1000);
    } else if (minAngle != -1 and maxAngle != -1) {
      calibration = false;
      digitalWrite(ONBOARD_LED, 0);
    }
  }
  // Serials handling
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    char e = Serial.read();
    if (msg == "BON") {
      btEn = true;
    } else if (msg == "BOFF") {
      btEn = false;
    } else if (msg == "SON") {
      srEn = true;
    } else if (msg == "SOFF") {
      srEn = false;
    } else if (msg == "T") {
      test = !test;
    } else if (msg == "CAL") {
      calibration = true;
      minAngle = -1;
      maxAngle = -1;
      Serial.println("OK");
    } else if (msg == "STATUS") {
      sprintf(buffer, "srEn: %d\tTest: %d\tCalibration: %d", srEn, test, calibration);
      Serial.println(buffer);
    } else if (msg == "1" || msg == "2" || msg == "3" || msg == "4" || msg == "5" || msg == "6" || msg == "7" || msg == "8" || msg == "9" || msg == "10") {
      poseProcessing(atoi(msg.c_str()));
    } else {
      Serial.println(msg);
    }

    if (!test) {
      digitalWrite(ONBOARD_LED, srEn);
    }
  }

  // Bluetooth serial commands
  /*
    if (btSerial.available() > 0)
    {
        String msg = btSerial.readStringUntil('\n');
        if (msg == "BTON")
        {
            btEn = true;
        }
        else if (msg == "BTOFF")
        {
            btEn = false;
        }
        else if (msg == "SERON")
        {
            srEn = true;
        }
        else if (msg == "SEROFF")
        {
            srEn = false;
        }
        else if (msg == "T")
        {
            test = !test;
        }
        else
        {
            btSerial.print(msg + '\n');
        }
        digitalWrite(ONBOARD_LED, btEn || srEn);
    }
    */
}