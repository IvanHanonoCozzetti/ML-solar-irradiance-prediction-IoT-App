# Machne Learning Solar Irradiance Prediction IoT Application
IoT web application that predicts solar irradiance by using Machine Learning algorithms in Python: Raspberry Pi Pico &amp; OpenWeatherMap API.<br>
**Ivan Hanono Cozzetti, ih222sf**

#### This is a description and tutorial on building a Machine Learning solar irradiance predictor, using real-time data gathered from a Raspberry Pi Pico WH and [OpenWeatherMap](https://openweathermap.org/) API.

The predictions made here are for [Direct Solar Irradiance](https://globalsolaratlas.info/support/faq) (also known as Beam Radiation).<br>
Note that this differs from diffuse solar radiation. This means that the prediction represents the solar irradiance on a direct surface.<br>
Meteorological data was gathered from [HI-SEAS weather station](https://www.hi-seas.org/), which lasted 4 months and was collected from an area in Hawaii with similar conditions to Mars. 

| Raspberry Pi Pico gathered data:  |OpenWeatherMap API data: |
|---|---|
| Humidity | Wind Speed |  
| Temperature | Wind Direction  | 
| Light | Air Pressure |  

In order to predict solar irradiance, the application runs input data read from the Raspberry Pi Pico and the API, through a number of algorithms built on 32.686 rows of data.

*How much time Will take to build this project?*<br>
That will highly depend on the environment and operating system you are using.<br>
If you have worked with Linux before, I highly suggest doing it so in a Linux machine (anything Debian-based should be easy, such as Ubuntu).<br>
Considering that, it should take between 2 to 5 hours to set everything up, depending on your experience working with programming environments, Linux, troubleshooting, and dependencies. 

## Objective
*Why this project?*<br>
The main motivation was to try to build something complex, using Machine Learning, out of simple and available sensors and data.<br>
Although this project is specifically set up to predict solar irradiance, a similar approach and steps could be taken to make other types of predictions: if you have other sensors, and
you are using a different dataset, for example from [Kaggle](https://www.kaggle.com/) (that has more than 350.000 open datasets available), then you could twitch hyperparameters and settings to
make your own predictive method.<br>
This is of course a simplification of the project. If you wish to fully understand what, how, and why, you should learn and understand many complex concepts. However, is a good start, and more importantly **motivating**.

*What's the purpose of this project?* <br>
The main purpose of this project is to predict solar irradiance out of real-time data, from multiple sensors from an RP Pico WH and APIs.<br>
Solar irradiance is not something that anybody measures daily for any unspecific need, but rather a scientific measure (kilowatts/watts per square meter).<br>
So there is no general purpose, however, this could help in both scientific and non-scientific applications:
1. A somehow general application for it could be the set-up and correction of solar panels' directions and angles: <br>
If you own solar panels, and you wish to correct or improve the efficiency of the irradiance over the panels' surface, you can use this IoT approach.<br>
Furthermore, one could automate the panel's angles based on the data fetched and the response from the ML models (which is a key advantage of working with full-duplex transmission).<br>
In practice, a device exists called *Solar Irradiance Meter* to know solar irradiance, however, these are very expensive (anywhere from 400 to 700 U.S. dollars).
2. In the scientific research area, one could use instead a dataset that is based on Diffuse Solar Radiation and set many of these IoT devices throughout an area (or entire country).<br>
This is the case because solar irradiation is known to be regionally consistent: on average, solar irradiation [does not change between 100km to 200km of distance](https://www.youtube.com/watch?v=cok7xtKnvV0). <br> 
Then, the data gathered could help for investigation and research of many different aspects: from weather, to irradiance and radiation impact, to state changes. <br>


*What insights will this project give?*  <br>
This project does cover and provide good insight into a wide range of areas:
- SkLeran widely used model and algorithms: How to set them up, set hyperparameters, learn about fitting and predicting (in this case, for a Regression problem)
- StreamLit: An open-source framework to build and deploy web applications, which is very good for data-based applications, with advanced data science tools, such as interactive plots.
- Connectivity with Mosquitto & Umqtt
- Implementation of asynchronous programming and threads: Implemented between real-time data display, ML prediction computation, display of predictions, and MQTT publishing.<br>
  This also includes basic utilization of `_threads` for the RP Pico, which is simple to implement, but highly valuable to achieve parallelism on the Pico as a full-duplex transmitter (acting as publisher with core 0 and subscriber with core 1).
- Open source feeling: If noticed, everything used to implement and launch this project is open source. StreamLit, Mosquitto/MQTT, Linux, and even OpenWeatherMap API (if I'm not mistaken).
I would consider this to be a key point, as some of the breakthroughs in the computer science and engineering world are achieved due to open-source existence. <br>


## Materials
<!---TODO: ADD IMAGES-------------------------------------------------------------------------------------------------------------)-->
| Item Name and Model | Price(SEK)/Price(EUR) | Seller | Image |
|---|---|---|---|
| 1. Raspberry Pi Pico WH | 109  /  9,60| [ElectroKit](https://www.electrokit.com/raspberry-pi-pico-wh) | |
| 2. Digital temperature and humidity sensor DHT11 | 49  /  4,30 | [ElectroKit](https://www.electrokit.com/digital-temperatur-och-fuktsensor-dht11) | | 
| 3. Photoresistor CdS 4-7 kohm | 8  /  0,70 | [ElectroKit](https://www.electrokit.com/fotomotstand-cds-4-7-kohm)| | 
| 4. LED 5mm 1500mcd (x9, green, yellow & red) | 45  /  4,00 | [ElectroKit](https://www.electrokit.com/led-5mm-rod-diffus-1500mcd) | | 
| 5. Jumper Cables (x20) | 49  /  4,30 | [ElectroKit](https://www.electrokit.com/labbsladd-40-pin-30cm-hane/hane) | | 
| 6. Resistors 100 ohm to 330 ohm (x10) | 10  /  0,90 | [ElectroKit](https://www.electrokit.com/motstand-kolfilm-0.25w-330ohm-330r) | | 
| 7. Resistor 10 kohm (x1) | 1  /  0,09 | [ElectroKit](https://www.electrokit.com/motstand-kolfilm-0.25w-10kohm-10k) | | 
| 8. USB cable A male - micro B male | 39  /  3,50 | [ElectroKit](https://www.electrokit.com/usb-kabel-a-hane-micro-b-5p-hane-1.8m) | | 


1. The Raspberry Pi Pico WH will be used as the client publishing the data captured from the sensors. In addition, it will also subscribe to the predictions, and reflect them through the LEDs based on the irradiance level. <br>
The pico is dual-core with 264kB internal RAM, and we will use these cores individually, hence, being a good option for this project. In addition, it has a 2.4GHz 802.11n wireless LAN module, which will be used for communicating in the network.
2. The DHT11 is a key sensor, as it will be used to detect both temperature and humidity.
3. The photoresistor is a good start to integrate light reflected over the surface, which is one of the key features when considering direct solar irradiance.<br>
Although the dataset does not have this feature, we will use it as a supporting feature (used a weight, rather than a feature).<br>
4. The LEDs represent a 9-level scale from the first one (low irradiance, 0 k/m^2 to 200 k/m^2) to nine (high irradiance, 1601 k/m^2 to 1800 k/m^2).

5.6.7.8. All these components are needed to connect and set the sensors and LEDs with the Pico's GPIO pins.

## Computer setup
To better organize instructions, I will here explain how to set the IDEs used, libraries, connectivity installations and everything else needed regarding setup.

### IDEs
#### Setting up Thonny (Linux machine or Linux VM, Debian-based OS):<be>
1. Open a new terminal
2. Enter the command `sudo apt install thonny`
3. Pres 'Y' each time you are prompted/asked to confirm a download/installation
That's it! You should be able to start Thonny from your installed programs.<br>
After opening it, you only need to select your preferred language (for the sake of further instructions, select English)
   
#### Setting up Visual Studio Code (Linux machine or Linux VM, Debian-based OS):<br>
1. Open [VScode website](https://code.visualstudio.com/) and click on the Download button on the top right of the screen.
2. Select `.deb` file to start the download.
3. Open a new terminal window and move to the Downloads directory by entering `cd Downloads`.
4. Install VScode by writing the command `sudo dpkg -i code_1.90...amd64.deb` (you need to write `sudo dpkg -i` + the name of the downloaded file in step 2, which is a `.deb` file).
Visual Studio Code should be installed after that.

#### Mosquitto Installation
1. Simply run the following commands in a new terminal<br>
`sudo apt install mosquitto`<br>
`sudo apt-get update`<br>
2. To verify the installation you can run mosquitto on the same terminal by writing `mosquitto`.
3. Then stop the terminal if it is already running (and run `sudo systemctl stop mosquitto & sudo pkill mosquitto` to verify it isn't running)

#### Mosquitto .conf File
You will need to modify the mosquitto configuration file with the IP address of the device being the broker:
1. Open a terminal and run `ifconfig` and find the IP for that device
2. Locate the conf file for mosquitto. It should be by default at: `/etc/mosquitto/mosquitto.conf`. You can find it by running `find /etc/mosquitto -name mosquitto.conf` on a terminal.
3. Once you find the location, we will edit that file with the information below, by writing the following command `sudo nano /etc/mosquitto/mosquitto.conf` and pasting the settings below:
```
# Place your local configuration in /etc/mosquitto/conf.d/
#
# A full description of the configuration file is at
# /usr/share/doc/mosquitto/examples/mosquitto.conf.example
allow_anonymous true
pid_file /run/mosquitto/mosquitto.pid

persistence true
persistence_location /var/lib/mosquitto/

log_dest file /var/log/mosquitto/mosquitto.log

include_dir /etc/mosquitto/conf.d
# ADDED TO LISTEN TO OTHER DEVICES IN THE NETWORK
listener 1883 YOUR-BROKER-IP
```
*NOTE: Replace YOUR-BROKER-IP with the device IP you will use as a broker, i.e. 192.168.xxx.xxx*

#### Raspberry Pi Pico software setup (firmware, MicroPython and MQTT)
**Firmware & MicroPython**
1. Connect the USB to the Pico.
2. Then press and hold the *bootsel* white button right next to the USB connection.
3. While still holding the *bootsel* button, connect the Pico to your Machine's USB port (and let go of the bootsel button).<br>
- If you are using a Linux machine, the device should be now recognized.<br>
- If you are using a Virtual Machine: Open VirtualBox and click on the Settings button on the top right -> Go to USB ->  On the right, click on the `+` icon to add a new USB and select the Raspberry Pi Pico. Then press the OK button (you may need to re-do the 3 steps before for the VM to recognize it).
4. Open Thonny. On the top, click the `Run` drop-down menu and select `Configure interpreter...`
5. On the top *Which kind of interpreter should Thonny use for running your code?* select `MicoPython (Raspberry Pi Pico)`
6. At the bottom right click on *Install or Update MicoPython*
7. Select the following settings:
  - Target volume: `RPI-RP2`
  - MicoPython variant: `Raspberry Pi Pico WH`
  - Click **Install**
  - Close all windows once it is **Done**, disconnect the USB and connect it again.
  - *Note, on a Virtual Machine you will need to repeat the process of adding a new `+` USB device, which will now called MicoPython board in FS mode, or something around those words. Then, in VM you may need again to disconnect and connect the device once again*.
To verify the Pico is recognized, you should click on `Local Python 3` on the bottom right and change it to `MicoPython (Raspberry Pi Pico)`

**Mqtt**
1. Open Thonny
2. Click on TOols top right
3. Manage Plug-ins
4. Write on the search bar umqtt.simple and click on `micopython.umqtt.simple` and click install

**Mqtt Alternative**<br>
If the above throws an error, you can try this:
1. Open this [link](https://pypi.org/project/micropython-umqtt.simple/#files) (https://pypi.org/project/micropython-umqtt.simple/#files).
2. Download `micropython-umqtt.simple-1.3.4.tar.gz` by clicking on that link.
3. On the Downloads folder, rick click, and "extract".
4. Open Thonny, click on the top bar **view**, and select **files**.
5. On the left, you should see your local machine files and the Raspberry Pi Pico files.
6. If no library called "lib" exists within the pico files, create a new directory with that name.
7. Then, from the extracted umqtt folder, open it and copy the *umqtt* folder inside the *lib* directory.

#### Python Libraries 
Python should be installed on your machine. To verify this, run `python3 --version` on a terminal (otherwise install it from the [Python website](https://www.python.org/downloads/), this is a simple installation). <br>
**NOTE:** You must have `Python 3.9` **or higher** Installed, as some code implementations in the thread run use methods only available since that version.<br>
1. Clone the project from GitHub into the directory you with to have the project `git clone https://github.com/IvanHanonoCozzetti/ML-solar-irradiance-prediction-IoT-App.git`
2. On a terminal, within the project's directory, run `pip install -r requirements.txt`<br>
This will install:
  - asyncio
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - requests
  - scikit-learn
  - plotly

#### NodeJS
1. Open a terminal and run the commands below:
   - `sudo apt update`
   - `sudo apt install nodejs`
   - `sudo apt install npm`
2. Verify nodeJS installation by `node -v` and `npm -v` for the npm.


### Putting everything together and pinout
The connections can be done as follows (the actual pinout in the images below slightly differs due to slight limitations at Wokwi, and different breadboard settings).
The important points are properly recognizing the Ground, 3.3V, and output reading pins, to avoid burning any components.<br>
**Note that ideally** you want to set the photoresistor the furthest away from the LEDs, such that the light generated from the LEDs does not change the light reading data.<br>
In the images below, we do not really see the "most" optimal position (the most optimal position is the photoresist on the further left corner, and the LEDs a bit closer to the Pico).
- The LED's cathode (negative) is the "short leg", while the anode (positive) is the "long leg".<br>
  The negative side can be directly connected to the ground rail (-) or through a jumper cable. The positive should be connected with a resistor (between 100ohm to 330ohm).<br>
  The opposite side of the resistors should be connected to the Pico's pins through a jumper cable.<br>
  The GPIOs used for the LEDs are 7 to 15 (the higher the GPIO, the higher the irradiance, the "redder" the LED should be, in series).<br>
  From the Pico's side, a connection between the negative rail (`-` in blue) and any of the ground pins should be done.<br>
- For the Photoresistor, one leg should be connected directly to a GPIO, in this case, GPIO 21.<br>
  The other end should be connected to a 10kohm resistor, and the resistor connected to the positive rail.<br>
  The positive rail, on the Pico's side, must be connected to the 3v3 OUT (output) pin, which is pin number 36.
- The Digital temperature and humidity sensor DHT11 has 3 pins:
  1. The most left pin (from the side shown both in the WokWI emulator and in the actual images, with a GREEN jumper), is the data reading pin: it connects to ADC 2 (which is GP28, which is pin number 34).
  2. The center pin connects to the positive rail (red jumper), which is connected on the Pico's side to the 3v3 OUT (output) pin.
  3. The most right side pin (connected with a black jumper) connects to the ground rail, connected on the Pico's end to any GND pin.
     
#### Pinout Emulator (better detail on the connections)
*Note: Positive and negative rails are inverted to the one shown in the actual connection.*
![pinout](images/pinout.png)

#### Actual Pinout used (alternative, better space management)
*NOTE: Positive and negative rails are inverted to the one shown in the emulator.*<br>
*Pinout example 1*
![actual_pinout](images/actual_pinout1.jpg)
*Pinout example 2*
![actual_pinout1](images/actual_pinout.jpg)
*Pinout example 3*
![actual_pinout2](images/actual_pinout2.jpg)
