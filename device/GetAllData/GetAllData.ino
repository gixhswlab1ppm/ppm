#include <MPU6050_tockn.h>
#include <Wire.h>

MPU6050 mpu6050(Wire);

long timer = 0;


void setup() {
  pinMode(7, INPUT);
  Serial.begin(9600);
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(false);
}

void transmit(packet &p){
  Serial.print('[')
  Serial.print(p.ts); Serial.print(',');
  Serial.print(p.accx); Serial.print(',');
  Serial.print(p.accy); Serial.print(',');
  Serial.print(p.accz; Serial.print(',');
  Serial.print(p.gyrox); Serial.print(',');
  Serial.print(p.gyroy); Serial.print(',');
  Serial.print(p.gyroz); Serial.print(',');
  Serial.print(p.a0); Serial.print(',');
  Serial.print(p.a1); Serial.print(',');
  Serial.print(p.a2); Serial.println(']');
}

void loop() {
  if (digitalRead(7) == 0){
    delay(1000);
    return;
  }

  // Serial.print('!');
  
  mpu6050.update();
  packet p;
  mpu6050.getPacket(&p);
  p.ts = millis();
  p.a0 = analogRead(A0);
  p.a1 = analogRead(A1);
  p.a2 = analogRead(A2);
  // Serial.write((byte*)&p, sizeof(p));
  // Serial.print('@');
  transmit(p);
  Serial.flush();
}

