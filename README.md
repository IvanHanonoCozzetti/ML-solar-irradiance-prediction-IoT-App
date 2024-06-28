# Machne Learning Solar Irradiance Prediction IoT Application
IoT web application that predicts solar irradiance by using Machine Learning algorithms in Python: Raspberry Pi Pico &amp; openweathermap API.<br>
**Ivan Hanono Cozzetti, ih222sf**

#### This is a description and tutorial on how to build a Machine Learning solar irradiance predictor, using real-time data gathered from a Raspberry Pi Pico WH and [openweathermap](https://openweathermap.org/) API.

The predictions made here are for [Direct Solar Irradiance](https://globalsolaratlas.info/support/faq) (also known as Beam Radiation).<br>
Note that this differs from diffuse solar radiation. This means that the prediction represents the solar irradiance in a direct surface.<br>
Meteorological data was gathered from [HI-SEAS weather station](https://www.hi-seas.org/), which lasted 4 months, and was gathered from an area in Hawaii with similar conditions to those in Mars. 

| Raspberry Pi Pico gathered data:  |OpenWeatherMap API data: |
|---|---|
| Humidity | Wind Speed |  
| Temperature | Wind Direction  | 
| Light | Air Pressure |  

In order to predict solar irradiance, the application runs input data read from the Raspberry Pi Pico and from the API, through a number of algorithms built on 32.686 rows of data.

*How much time it might take to build this projct?*<br>
That will higly depend on the enviorenment and operating system you are using.<br>
If you have worked with Linux before, I highly suggest doing it so in a Linux machine (anything Debian based should be easy, such as Ubuntu).<br>
Considering that, it should take between 2 to 5 hours to set everything up, depending on your experience working with programming environments, Linux, troubleshooting and dependencies. 

### Objective
*Why this project?*<br>
The main motivation was to try to build something complex, using Machine Learning, out of simple and available sensors and data.<br>
Although this project is specifically setted up to predict solar irradiance, a similar approach and steps could be taken to make other type of predictions: if you have other sensors, and
you are using a different dataset, for example from [Kaggle](https://www.kaggle.com/) (that has more than 350.000 open datasets available), then you could twich hyperparameters and settings to
make your own predictive method.<br>
This is of course a simplification of the project. If you wish to fully understand what, how and why, you should learn and understand many complex concepts. However, is a good start, and more importantly **motivating**.

*Whats the purpose of this project?* <br>
The main purpose of this project is to predict solar irradiance out of real-time data, from multiple sensors from a RP Pico WH and APIs.<br>
Solar irradiance is not something that anybody measures on the daily bases for any unespecific need, but rahter a scientific measure (kilowatts/watts per square meter).<br>
So there is not a general purpose, however, this could help in both scientific and non-scientific applications:
1. A somehow general application for it could be the set-up and correction of solar panels directions and angles: <br>
If you own solar panels, and you wish to be correct or improve efficiency of the irradiance over the panels' surfice, you can use this IoT approach.<br>
Furhtermore, one could automate the panel's angles based on the data fetched and the response from the ML models (which is a key adventage of working with full-duplex transmission).<br>
In practice, a dvice exists called *Solar Irradiance Meter* to know solar irradiance, however, these are very expensive (anywhere from 400 to 700 U.S. dollars).
2. In the scientific research area, one could use instead a dataset that is based on Diffuse Solar Radiation, and set many of these IoT devices throughout an area (or entire country).<br>
This is the case because solar irradiation is known to be regionally consistent: on average, solar irradiation [does not change between 100km to 200km of distance](https://www.youtube.com/watch?v=cok7xtKnvV0). <br> 
Then, the data gathered could help for investigation and researching of many different aspects: from weather, to irradiance and radiation impact, to state changes.<br>
