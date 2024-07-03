import dht
from machine import Pin   
from machine import ADC
import network
from umqtt.simple import MQTTClient
import asyncio
import _thread
import time

# Settings pin for DHT11 sensor
humid_temp_sensor = dht.DHT11(Pin(21))
photo_resistor = machine.ADC(2)
# Used to note when a new prediction request is sent
integrated_led = Pin("LED", Pin.OUT)

# WiFi Settings: name/ssid and password
WIFI_SSID = "your wifi's name"
WIFI_PASSWORD = "your wifi's password"

# MQTT client with broker configuration
MQTT_BROKER = "192.168.xxx.xxx"
MQTT_PORT = 1883
MQTT_TOPIC = "HumidTempPredict"
MQTT_TOPIC2 = "HumidTempRealTime"
MQTT_TOPIC_SUB = "PredictionResults"

client = MQTTClient("Pico_Client", MQTT_BROKER, MQTT_PORT)
client_listener = MQTTClient("Pico_Listener", MQTT_BROKER, MQTT_PORT)

# This dictionary is used to match irradiance prediction to a "level" which is the LED Gpio to turn on 
irradiance_level_led = {
    (0,200):7,
    (201,400):8,
    (401,600):9,
    (601,800):10,
    (801,1000):11,
    (1001,1200):12,
    (1201,1400):13,
    (1401,1600):14,
    (1601,1800):15,
}

# Establishing wifi network
def connect_to_wifi():
    # Wifi (station interface), activation and connection with set credentials
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    while wlan.isconnected() == False:
        print('Waiting for connection......')
        time.sleep(1)
    print("+Connected to Wifi!")


# Connecting to MQTT broker
def connect_to_mqtt(client_to_connect, client_name):
    # Actual connection to broker
    client_to_connect.connect()
    print("+Connected to MQTT Broker!", client_name)


# Reading humidity and temperature from I/O, printing it and returning them
def read_humd_temp():
    try:
        humid_temp_sensor.measure()
        temperature = humid_temp_sensor.temperature()
        humidity = humid_temp_sensor.humidity()
        brightness = photo_resistor.read_u16()
        print("Temperature is {} degrees Celsius, Humidity is {}% and brightness is {}.".format(temperature, humidity, brightness))
        return humidity, temperature, brightness
    except:
        print("Error in reading sensor values")


# Making a prediction request
def publish_humd_temp_predict():
    humidity, temperature, brightness = read_humd_temp()
    client.publish(MQTT_TOPIC, str(humidity) + "," + str(temperature) + "," + str(brightness))

# Publishing humidity and temperature to broker
def publish_humd_temp_real_time():
    humidity, temperature, brightness = read_humd_temp()
    client.publish(MQTT_TOPIC2, str(humidity) + "," + str(temperature) + "," + str(brightness))


# Publish function (publishes data via a thread)
def publish_thread():
    start = time.time()
    while True:
        current_time = time.time()
        publish_humd_temp_real_time()
        # This the time in second that each new prediction request is sent
        if (current_time - start) >= 30:
            integrated_led.on()  # Turns on integrated LED each time a prediction is requested
            # time.sleep(0.5)
            # Publish temperature data and wait 30 seconds (which is about the time the ML models take to predict)
            print("\n>>>Sending Prediction Request......\n")
            publish_humd_temp_predict()
            time.sleep(4)
            start = current_time
            integrated_led.off()
        time.sleep(0.5)

def callback_prediction(topic, msg):
    msg_decoded = msg.decode('utf-8')
    predicted_val = float(msg_decoded)
    print("\n-----> The solar irradiance predicted was:", predicted_val, "\n")
    # Based on the 'irradiance_level_led' dictionary, we check the tuple val, and if in range assign it to that LED Gpio to turn it on
    for key, value in irradiance_level_led.items():
        if key[0] <= predicted_val <= key[1]:
            led=Pin(value, Pin.OUT)
            led.on()
            time.sleep(5)
            led.off()

# Reads predictions made
def subscriber_thread():
    try:
        while True:
            print("<<<Waiting & Listening to prediction results>>>")
            client_listener.wait_msg()
    except Exception as e:
        print(f"An error happend while waiting for the message {e}")
    # finally
        # client_listener.disconnect()


def main():
    # Connect to Wifi
    print("Connecting to Wifi......")
    connect_to_wifi()
    print("Connecting to MQTT......")
    connect_to_mqtt(client, "(publisher!)")
    connect_to_mqtt(client_listener, "(subscriber!)")
    
    client_listener.set_callback(callback_prediction)
    client_listener.connect()
    client_listener.subscribe("PredictionResults")  # which in Linux terminal: mosquitto_sub -h '192.168.xxx.xxx' -t 'PredictionResults'
    print("+Subscribed to prediction results!")

    # Run subscriber thread in core 1
    sub_thread = _thread.start_new_thread(subscriber_thread, ())  # <- This start_new_thread take tuples, for parameters and so on, hence the ().
    # Run publisher thread in core 0
    publish_thread()
    
main()