/*
  Knock Sensor

  This sketch reads a piezo element to detect a knocking sound.
  It reads an analog pin and compares the result to a set threshold.
  If the result is greater than the threshold, it writes "knock" to the serial
  port, and toggles the LED on pin 13.

  The circuit:
  - positive connection of the piezo attached to analog in 0
  - negative connection of the piezo attached to ground
  - 1 megohm resistor attached from analog in 0 to ground

  created 25 Mar 2007
  by David Cuartielles <http://www.0j0.org>
  modified 30 Aug 2011
  by Tom Igoe

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/Knock
*/


// these constants won't change:
const int ledPin = 13;      // LED connected to digital pin 13
const int knockSensor_1 = A0; // the piezo is connected to analog pin 0
const int knockSensor_2 = A1; // the piezo is connected to analog pin 0
const int knockSensor_3 = A2; // the piezo is connected to analog pin 0
const int threshold = 10;  // threshold value to decide when the detected sound is a knock or not


// these variables will change:
int sensorReading_1 = 0;      // variable to store the value read from the sensor pin
int sensorReading_2 = 0;      // variable to store the value read from the sensor pin
int sensorReading_3 = 0;      // variable to store the value read from the sensor pin
int ledState = LOW;         // variable used to store the last LED status, to toggle the light

void setup() {
  pinMode(ledPin, OUTPUT); // declare the ledPin as as OUTPUT
  Serial.begin(9600);       // use the serial port
}

void loop() {
  // read the sensor and store it in the variable sensorReading:
  sensorReading_1 = analogRead(knockSensor_1);
  if (sensorReading_1 > threshold) {
    Serial.println("sensor 1:");
    Serial.println(sensorReading_1);
  }
  sensorReading_2 = analogRead(knockSensor_2);
  if (sensorReading_2 > threshold) {
    Serial.println("sensor 2:");
    Serial.println(sensorReading_2);
  }
  sensorReading_3 = analogRead(knockSensor_3);
  if (sensorReading_3 > threshold) {
    Serial.println("sensor 3:");
    Serial.println(sensorReading_3);
  }
  // if the sensor reading is greater than the threshold:
  if (sensorReading_1 + sensorReading_2 + sensorReading_3 >= (threshold * 3)) {
    // toggle the status of the ledPin:
    ledState = !ledState;
    // update the LED pin itself:
    digitalWrite(ledPin, ledState);
    // send the string "Knock!" back to the computer, followed by newline
    Serial.println("Knock!");
    Serial.println("Estimate: ");
    if (sensorReading_3 < sensorReading_1) {
      if (sensorReading_3 < sensorReading_2) {
        Serial.println("Ball in between 1 and 2");
      }
    }
    if (sensorReading_2 < sensorReading_1) {
      if (sensorReading_2 < sensorReading_3) {
        Serial.println("Ball in between 1 and 3");
      }
    }
    if (sensorReading_1 < sensorReading_3) {
      if (sensorReading_1 < sensorReading_2) {
        Serial.println("Ball in between 3 and 2");
      }
    }
  }
  delay(100);  // delay to avoid overloading the serial port buffer
}
