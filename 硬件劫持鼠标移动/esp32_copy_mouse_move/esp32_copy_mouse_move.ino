#include <Arduino.h>
#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

void setup() {
    Serial.begin(115200);
    delay(100);

    USB.begin();  //把ESP32作为USB设备
    Mouse.begin();  //同时作为HID鼠标设备
}

void loop() {

    //检查是否有数据
    if (Serial.available() > 0) {

        //
        String data = Serial.readStringUntil('\n');
        data.trim();
        int comma = data.indexOf(',');

        if (comma != -1) {

            //x轴偏移量（相对位移）
            int ex = data.substring(0, comma).toInt();

            //y轴偏移量（相对位移）
            int ey = data.substring(comma + 1).toInt();

            // 执行鼠标相对移动
            Mouse.move(ex, ey);
        }
    }
    delay(1);
}
