"""
GVInput HID 设备 Python 封装
使用 pywinusb 发送相对鼠标移动
"""
import pywinusb.hid as hid


def clamp(v):
    return max(-127, min(127, int(v)))


class GVInputMouse:
    def __init__(self):
        devices = hid.HidDeviceFilter(
            vendor_id=0x00FF, product_id=0xBACC
        ).get_devices()
        if not devices:
            raise RuntimeError(
                "未找到 GVInput 设备 (VID_00FF/PID_BACC)。\n"
                "请确认已安装 UU远程/GameViewer。"
            )
        self.device = devices[0]
        self.device.open()
        print("✓ GVInput 设备已打开")

    def send_report(self, inner_report):
        report = [0x40] * 65
        report[0] = 0x40
        report[1] = len(inner_report)
        for i, b in enumerate(inner_report):
            report[2 + i] = b & 0xFF
        self.device.send_output_report(report)

    def move_relative(self, dx, dy, wheel=0):
        inner = [
            0x04,
            0x00,
            clamp(dx) & 0xFF,
            clamp(dy) & 0xFF,
            clamp(wheel) & 0xFF,
        ]
        self.send_report(inner)

    def close(self):
        if self.device:
            self.device.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import time
    try:
        mouse = GVInputMouse()
        print("测试: dx=50, dy=0")
        mouse.move_relative(50, 0)
        time.sleep(0.5)
        print("测试: dx=-50, dy=0")
        mouse.move_relative(-50, 0)
        print("✓ 完成，如果鼠标动了说明 GVInput 工作正常")
    except Exception as e:
        print(f"✗ {e}")
