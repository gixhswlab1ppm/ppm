#include <ESP8266WiFi.h>
#include <FirebaseArduino.h>
//#include <SoftwareSerial.h>
#define FIREBASE_HOST "hwsw-lab.firebaseio.com"

#define FIREBASE_AUTH "pykuEBabbXC50qJOz4iOrsi9AMPT5xLAiZSwIDXI"

//SoftwareSerial esp8266(2,
#define WIFI_SSID "shine"
#define WIFI_PASSWORD "11111111"


#define LED 2
char buf[12]={0};
bool is_first = true;
String readstring;
void setup() {

  pinMode(LED, OUTPUT);
  digitalWrite(LED, 0);
  Serial.begin(9600);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("connecting");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.println();
  Serial.print("connected: ");
  Serial.println(WiFi.localIP());
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
  //Firebase.setInt("LEDStatus", 0);
  Firebase.setString("Nano", " ");


}

void loop() {
  int index = 0;
  while (Serial.available()) {
    if (Serial.read() == '\n') {
      if (is_first) {
        is_first = false;
        continue;
      }
      else {
        Serial.println(buf);
        Firebase.setString("Nano", buf);
        memset(buf,0,sizeof(buf));
        index = 0;
      }
    }
    else {
      buf[index] = Serial.read();
      index = index + 1;
    }
  }


  if (Firebase.failed()) // Check for errors {
  {
    Serial.print("setting /number failed:");
    Serial.println(Firebase.error());
    return;
  }

  delay(1000);

}
