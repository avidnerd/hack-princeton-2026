// Spoonful GO button over Bluetooth: the ESP32 pairs with the laptop as a Bluetooth keyboard and "types" Page Down
// when the arcade button is pressed. The Spoonful web app treats Page Down (and Enter / Page Up) as GO.
//
// Why a keyboard: the laptop's only micro-USB cable is needed by the arm, and Windows Smart App Control blocks the
// Python Bluetooth library. A Bluetooth keyboard needs no software on the laptop at all.
//
// GO only. This is NOT a STOP input: STOP is voice "stop", spacebar, the on-screen STOP, looking away, or the arm's
// power switch. GO alone never moves the spoon toward the face: the face gate (looking + mouth open) must also pass.
//
// Board: ESP32-C3 (e.g. ESP32-C3-DevKitM-1). Arduino: board "ESP32C3 Dev Module", core esp32 3.x.
// Wiring: arcade button (COM / NO tabs) between BTN_PIN and GND. Internal pull-up, so pressed = LOW.
// Power: 5 V into the board's 5V pin from a breadboard power module, or the micro-USB port (never both at once).
// Pair once: Windows Settings > Bluetooth & devices > Add device > Bluetooth > "Spoonful GO".
// The keyboard types into the window in front: keep the Spoonful browser tab focused during a session.

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEHIDDevice.h>
#include <BLESecurity.h>
#include <HIDTypes.h>

const int BTN_PIN = 4;
const uint8_t KEY_PAGE_DOWN = 0x4E;   // USB HID usage id
const uint32_t DEBOUNCE_MS = 25;
const uint32_t MIN_GAP_MS = 300;      // ignore bounces / double taps closer than this

// Standard keyboard report (report id 1): modifiers, reserved, 6 key codes.
static const uint8_t REPORT_MAP[] = {
  0x05, 0x01, 0x09, 0x06, 0xA1, 0x01, 0x85, 0x01,
  0x05, 0x07, 0x19, 0xE0, 0x29, 0xE7, 0x15, 0x00, 0x25, 0x01, 0x75, 0x01, 0x95, 0x08, 0x81, 0x02,
  0x95, 0x01, 0x75, 0x08, 0x81, 0x01,
  0x95, 0x06, 0x75, 0x08, 0x15, 0x00, 0x25, 0x65, 0x05, 0x07, 0x19, 0x00, 0x29, 0x65, 0x81, 0x00,
  0xC0
};

BLEHIDDevice* hid = nullptr;
BLECharacteristic* input = nullptr;
volatile bool connected = false;
bool btnStable = false, btnRaw = false;
uint32_t lastChange = 0, lastSent = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer*) override { connected = true; Serial.println("{\"ble\":\"connected\"}"); }
  void onDisconnect(BLEServer*) override {
    connected = false;
    Serial.println("{\"ble\":\"disconnected\"}");
    BLEDevice::startAdvertising();   // let the laptop reconnect
  }
};

void sendKey(uint8_t key) {
  uint8_t down[8] = {0, 0, key, 0, 0, 0, 0, 0};
  uint8_t up[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  input->setValue(down, sizeof(down));
  input->notify();
  delay(15);
  input->setValue(up, sizeof(up));
  input->notify();
}

void setup() {
  Serial.begin(115200);
  pinMode(BTN_PIN, INPUT_PULLUP);

  BLEDevice::init("Spoonful GO");
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  hid = new BLEHIDDevice(server);
  input = hid->inputReport(1);
  hid->pnp(0x02, 0xe502, 0xa111, 0x0210);
  hid->hidInfo(0x00, 0x01);
  hid->reportMap((uint8_t*)REPORT_MAP, sizeof(REPORT_MAP));
  hid->startServices();
  hid->setBatteryLevel(100);

  BLESecurity* security = new BLESecurity();
  security->setAuthenticationMode(ESP_LE_AUTH_BOND);

  BLEAdvertising* adv = server->getAdvertising();
  adv->setAppearance(HID_KEYBOARD);
  adv->addServiceUUID(hid->hidService()->getUUID());
  adv->start();
  Serial.println("{\"ble\":\"advertising\",\"name\":\"Spoonful GO\"}");
}

void loop() {
  uint32_t now = millis();
  bool pressed = digitalRead(BTN_PIN) == LOW;
  if (pressed != btnRaw) { btnRaw = pressed; lastChange = now; }
  if (btnRaw != btnStable && now - lastChange >= DEBOUNCE_MS) {
    btnStable = btnRaw;
    Serial.println(btnStable ? "{\"btn\":1}" : "{\"btn\":0}");
    if (btnStable && now - lastSent >= MIN_GAP_MS) {
      lastSent = now;
      if (connected) {
        sendKey(KEY_PAGE_DOWN);
        Serial.println("{\"sent\":\"PageDown\"}");
      } else {
        Serial.println("{\"sent\":false,\"why\":\"not paired/connected\"}");
      }
    }
  }
  delay(2);
}
