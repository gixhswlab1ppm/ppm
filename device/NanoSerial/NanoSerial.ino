#include <SoftwareSerial.h>
SoftwareSerial myserial (4,5); //RX,TX

void setup() {
    Serial.begin(9600);
     while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  } 
  myserial.begin(9600);

}
char arr[20]="1,2,3,4,5,1,2,3";

void loop() {
  
  for(int i=0;i<sizeof(arr);i++){
      Serial.print(arr[i]);
  };
    Serial.print('\n');
//  while(myserial.available()>0){
//    Serial.print("Read from 1");
//    Serial.print(myserial.read());  
//    Serial.println();
//  }
  delay(500);
}
