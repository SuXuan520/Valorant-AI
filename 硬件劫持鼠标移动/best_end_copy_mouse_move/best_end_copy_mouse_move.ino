#include <Arduino.h>
#include "USB.h"
#include "USBHIDMouse.h"
       
USBHIDMouse Mouse;

//CH9350L帧解析

#define CH9350_SERIAL  Serial2

// 帧头
#define CH_HEAD1  0x57
#define CH_HEAD2  0xAB

// 帧类型标识
#define CH_TYPE_HID_1     0x88
#define CH_TYPE_HID_2     0x0B

#define CH_HID_DATA_LEN   11

#define MOUSE_BTN_OFFSET   2
#define MOUSE_X_OFFSET     3
#define MOUSE_Y_OFFSET     5
#define MOUSE_WHEEL_OFFSET 7

// 帧解析状态机
enum Ch9350State {
    WAIT_HEAD1,
    WAIT_HEAD2,
    WAIT_TYPE1,
    WAIT_TYPE2,
    READ_HID_DATA,
    READ_CHECKSUM,
};
Ch9350State ch_state = WAIT_HEAD1;
uint8_t ch_hid[CH_HID_DATA_LEN];
uint8_t ch_hid_idx = 0;
uint8_t ch_type1 = 0, ch_type2 = 0;



//返回 true 表示收到完整一帧 HID 数据
bool ch9350_feed(uint8_t b) {
    switch (ch_state) {
        case WAIT_HEAD1:
            if (b == CH_HEAD1) { ch_state = WAIT_HEAD2; }
            break;
        case WAIT_HEAD2:
            ch_state = (b == CH_HEAD2) ? WAIT_TYPE1 : WAIT_HEAD1;
            break;
        case WAIT_TYPE1:
            ch_type1 = b;
            ch_state = WAIT_TYPE2;
            break;
        case WAIT_TYPE2:
            ch_type2 = b;
            if (ch_type1 == CH_TYPE_HID_1 && ch_type2 == CH_TYPE_HID_2) {
                ch_hid_idx = 0;
                ch_state = READ_HID_DATA;
            } else {
                ch_state = WAIT_HEAD1;   // 非 HID 帧直接丢弃
            }
            break;
        case READ_HID_DATA:
            ch_hid[ch_hid_idx++] = b;
            if (ch_hid_idx >= CH_HID_DATA_LEN) {
                ch_state = READ_CHECKSUM;
            }
            break;
        case READ_CHECKSUM:
            ch_state = WAIT_HEAD1;       // 校验字节跳过，回到帧头等待
            return true;                 // 通知：完整一帧就绪
    }
    return false;
}



//初始化
void setup() {
    Serial.begin(115200);                              // Python 通信串口
    delay(100);

    CH9350_SERIAL.begin(115200, SERIAL_8N1, 17);      // CH9350 串口 (RX=GPIO17)

    USB.begin();
    Mouse.begin();

    Serial.println("鼠标模拟已就绪");
}

//主循环
//位移倍率：原始数据*倍率，补偿CH9350L低速USB导致的 DPI 下降
//1.5 = 1.5 倍灵敏度，2.0 = 2 倍，1.0 = 原始。根据手感微调
#define MOUSE_SENSITIVITY 2.0f

void loop() {
    //倍率放大后超 ±127 的部分分片发送，不丢数据
    static int32_t carry_dx = 0, carry_dy = 0, carry_wheel = 0;

    //只取一帧，USB HID 单report队列排空多帧只会丢帧
    bool got_frame = false;
    int16_t mouse_dx = 0, mouse_dy = 0;
    int8_t  mouse_wheel = 0;
    uint8_t mouse_btn = 0;

    while (CH9350_SERIAL.available() > 0 && !got_frame) {
        uint8_t b = CH9350_SERIAL.read();
        if (ch9350_feed(b)) {
            mouse_dx    = (int16_t)(ch_hid[MOUSE_X_OFFSET] | (ch_hid[MOUSE_X_OFFSET + 1] << 8));
            mouse_dy    = (int16_t)(ch_hid[MOUSE_Y_OFFSET] | (ch_hid[MOUSE_Y_OFFSET + 1] << 8));
            mouse_wheel = (int8_t)ch_hid[MOUSE_WHEEL_OFFSET];
            mouse_btn   = ch_hid[MOUSE_BTN_OFFSET];
            got_frame = true;
        }
    }

    if (got_frame) {
        //按键响应
        if (mouse_btn & 0x01) Mouse.press(MOUSE_LEFT);
        else                  Mouse.release(MOUSE_LEFT);
        if (mouse_btn & 0x02) Mouse.press(MOUSE_RIGHT);
        else                  Mouse.release(MOUSE_RIGHT);
        if (mouse_btn & 0x04) Mouse.press(MOUSE_MIDDLE);
        else                  Mouse.release(MOUSE_MIDDLE);

        //乘倍率后累加，发送PC位移=物理位移*MOUSE_SENSITIVITY
        carry_dx    += (int32_t)(mouse_dx    * MOUSE_SENSITIVITY);
        carry_dy    += (int32_t)(mouse_dy    * MOUSE_SENSITIVITY);
        carry_wheel += (int32_t)(mouse_wheel * MOUSE_SENSITIVITY);
    }

    //每轮发一帧USBHID，剩余暂存carry
    int8_t sx = constrain(carry_dx,    -128, 127);
    int8_t sy = constrain(carry_dy,    -128, 127);
    int8_t sw = constrain(carry_wheel, -128, 127);

    if (sx != 0 || sy != 0) {
        Mouse.move(sx, sy);
        carry_dx -= sx;
        carry_dy -= sy;
    }
    if (sw != 0) {
        Mouse.move(0, 0, sw);
        carry_wheel -= sw;
    }

    //Python 自瞄数据
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil('\n');
        data.trim();
        int comma = data.indexOf(',');
        if (comma != -1) {
            int ex = data.substring(0, comma).toInt();
            int ey = data.substring(comma + 1).toInt();
            Mouse.move(ex, ey);
        }
    }
    delay(1);
}