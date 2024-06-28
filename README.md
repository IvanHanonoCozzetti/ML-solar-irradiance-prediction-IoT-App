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

### Objective
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


### Materials
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

