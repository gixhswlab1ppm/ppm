
#include <MPU6050_tockn.h>
#include <Wire.h>

MPU6050 mpu6050(Wire);

long timer = 0;


void setup() {
  pinMode(13, OUTPUT);
  Serial.begin(9600);
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(false);
}

void loop() {
  Serial.print('!');
  
  mpu6050.update();
  packet p;
  mpu6050.getPacket(&p);
  p.a0 = analogRead(A0);
  p.a1 = analogRead(A1);
  p.a2 = analogRead(A2);

  Serial.write((byte*)&p, sizeof(p));
  Serial.print('@');
  Serial.flush();
  
  // timer = millis();
   
}
