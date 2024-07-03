import asyncio
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import time
import requests
# from sklearn.linear_model import LinearRegression
# from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
import subprocess
# import threading
import plotly.express as px

# Loading CSV and dropping irrelevant columns
def load_data():
	df = pd.read_csv('solar_prediction.csv')
	df = df.drop(['UNIXTime', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
	df = df[df['Radiation']>0]
	return df

# Tab name
st.set_page_config(page_title="Solar Irradiance Prediction App") # page_icon=

# WEB APP CONTAINERS
# Title and Main
title_holder = st.empty()
inital_text = st.empty()
presentation_plot = st.empty()
presentation_end = st.empty()
predictions_title = st.empty()
give_param_placeholder = st.empty()
# Center Charts
title_plot_placeholder = st.empty()
bar_chart_placeholder = st.empty()
# RMSE and Pred table
table_pl = st.empty()
# Historical predictions 
text3_placeholder = st.empty()
scatter_predictions = st.empty()

# SIDE DATA
# Side plots (real time data)
plot1_title = st.sidebar.empty()
plot_slot1 = st.sidebar.empty()
plot2_title = st.sidebar.empty()
plot_slot2 = st.sidebar.empty()
plot3_title = st.sidebar.empty()
plot_slot3 = st.sidebar.empty()
# Heatmap
heatmap = st.sidebar.empty()
checkbox =st.sidebar.empty()
heatmap_slot =st.sidebar.empty()


df = load_data()
show_heatmap = checkbox.checkbox('Show Intercorrelation Heatmap')

if show_heatmap:
	df.to_csv('heatmap_data.csv', index=False)
	df_hm = pd.read_csv('heatmap_data.csv')
	corr = df_hm.corr()
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True
	with sns.axes_style("white"):
		figure, axis = plt.subplots(figsize=(7,5))
		axis = sns.heatmap(corr, mask=mask, vmax=1)
	heatmap_slot.pyplot(figure)
else:
	heatmap_slot.empty()


def update_parameters(curren_humidity_val, current_temp_farenheit):
	# Api data (2 out of 3 parameters, brigthness weight is calculated after the final prediction)
	url_base = "http://api.openweathermap.org/data/2.5/weather?"
	key = "your_api_key"
	city = "Vaxjo"
	full_url = url_base + "appid=" + key + "&q=" + city
	api_response = requests.get(full_url).json()
	api_pressure = api_response['main']['pressure']
	api_wind_direction = api_response['wind']['deg'] # degrees
	api_wind_speed = api_response['wind']['speed']
	params={'Temperature' : current_temp_farenheit,
		   	'Pressure' : api_pressure,
	      	'Humidity' : curren_humidity_val,
	      	'WindDirection(Degrees)' : api_wind_direction,
	      	'Speed' : api_wind_speed}
	return params

def web_layout():
	# Background
	title_holder.title('Solar :red[Irradiance] Prediction App')

	# Body
	inital_text.markdown('''
	This a realtime solar irradiance machine learning prediction application, which uses data gathered from a Raspberry Pi Pico WH and from OpenWeatherMap API.
	
	The predictions made here are for Direct Solar Irradiance (also known as Beam Radiation, and not diffuse solar radiation).
	Meteorological data was gathered from HI-SEAS weather station, which lasted 4 months, and was gathered from an area in Hawaii with similar conditions to those in Mars. 
	''')
	
	data = pd.read_csv('solar_prediction.csv')
	data = data.drop(data.index[0])
	fig = px.scatter(
		data,
		y="Radiation",
		color="Radiation",
		color_continuous_scale="reds",
	)
	presentation_plot.plotly_chart(fig, theme="streamlit", use_container_width=True)
	
	presentation_end.html('''
	Note that solar irradiance is a power per unit area measure, which is the surface power density, and the results will be expressed in
	watts per square meters (W/m^2) and not in kilowatts (Kw/m^2). <br>
	<span style="font-size: 20px">
	<pre>
	Raspberry Pi Pico gathered data:                OpenWeatherMap API data:
	- Humidity					- Wind Speed
	- Temperature					- Wind Direction
	- Light						- Air Pressure
	</pre>	   
	</span>
	In order to predict solar irradiance, the application will run input data read from the Raspberry Pi Pico and from the API, 
	through a number of algorithms based on 32.686 rows of data.
	''')

# These two are no longer allowed since we are using a Thread, but can be used to cache-in data, to save computation resources + lower exec. time
# @st.cache_data
# @st.cache_resource
def building_machine_learning_models(current_params):
	X = df[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']]
	y = df['Radiation']
	# Training data was only used to determine and improve performance, not really needed for each individual prediction
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
	models = [RandomForestRegressor(n_estimators=170, max_depth=25),
	          DecisionTreeRegressor(max_depth=30),  # More for decision tree feedback rather than predictions themselves
	          GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, max_depth=5),
			  KNeighborsRegressor(n_neighbors=7)]
			  # DummyRegressor(strategy='mean'), # Only used for testing, should not be used for real predictions
			  # ,LinearRegression(n_jobs=100)    # No real linear relationship between input and output data
	df_models = pd.DataFrame()
	temporary_hold = {}
	# Run through the models
	for model in models:
		print(model)
		# Conversting to string representation the models
		m = str(model)
		# Getting the name from the string representation (names are followed by parentheses, to be divded)
		temporary_hold['Model'] = m[:m.index('(')]
		model.fit(X_train, y_train)
		# RMSE (Root Mean Squared Error) with the training set will allow us to see the performance on the model before doing the prediction
		# Note that RMSE should not be calculated on a single data sample, but rather on a set (hence using the test set)
		temporary_hold['RMSE_Radiation'] = sqrt(mse(y_test, model.predict(X_test)))
		temporary_hold['Pred Value'] = model.predict(pd.DataFrame(current_params, index=[0]))[0]
		print('RMSE score', temporary_hold['RMSE_Radiation'],"\n")
		# Appending results from temporary into df_models data frame
		df_models = df_models._append([temporary_hold])
	# Setting each model column as the index for df_models
	df_models.set_index('Model', inplace=True)
	# .argmin gives us the model with lowest value in df_models on the column RMSE_Radiation, AKA the model with the lowest RMSE
	# .iloc find that model using the index
	# double [[]] is used to assure that we get a DataFrame, since with .values we conver it into a number array, as a float val.
	pred_value = df_models['Pred Value'].iloc[[df_models['RMSE_Radiation'].argmin()]].values.astype(float)
	return pred_value, df_models


def run_data(current_params):
	# Df models is now a tuple containing (pred_value, df_models)
	df_models = building_machine_learning_models(current_params)
	return df_models

def write_results(params, brightness, brightness_weight, current_temp_celsius, df_models, final_prediction):
	# Terminal print-outs
	print('Given the parameters:')
	print("Temperature:", current_temp_celsius, " Humidity:", params['Humidity'],
	   	  " Brightness:", brightness, " Pressure:", params['Pressure'], " WindDirection(Degrees)", params['WindDirection(Degrees)'],
		  " Speed:", params['Speed'])
	print("Solar radiation would be",final_prediction,"W/m2 (watts per square meter).\n\n")
	# Writing results into the web application
	humid = params['Humidity']
	press = params['Pressure']
	wind_dir = params['WindDirection(Degrees)']
	speed = params['Speed']

	predictions_title.markdown("### :red[Machine Learning Predictions]")
	give_param_placeholder.write(f'Given the parameters:\n'
                       f'- Temperature: {current_temp_celsius}\n'
                       f'- Humidity: {humid}\n'
					   f'- Brightness: {brightness}\n'
					   f'- Brightness Weight: {brightness_weight}\n'
					   f'- Models Predictions: {df_models}\n'
                       f'- Pressure: {press}\n'
                       f'- WindDirection(Degrees): {wind_dir}\n'
                       f'- Speed: {speed}\n\n'
                       f'Solar radiation would be {final_prediction} W/m^2 (watts per square meter).')
	
def show_ML(df_models, total_predictions, samples):
	title_plot_placeholder.write('**This diagram shows Root Mean Square Error for all models:**')
	bar_chart_placeholder.bar_chart(df_models['RMSE_Radiation'], color='#FF0000')

	table_pl.table(df_models)

	all_predictions = pd.DataFrame({
		'Radiation':total_predictions,
		'Prediction_Number':samples})
	
	text3_placeholder.write('**Prediction History:**')
	# Basic scatter plot
	# scatter_predictions.scatter_chart(total_predictions, color='#FF0000')
	# Nicer scatter in streamlit, requires many more parameters, but looks much better
	fig = px.scatter(
		all_predictions,
		x="Prediction_Number",
		y="Radiation",
		color="Radiation",
		color_continuous_scale="blues",
		size_max=15,
		size='Prediction_Number',
	)
	scatter_predictions.plotly_chart(fig, theme="streamlit", use_container_width=True)
	

async def predict_and_display():
	# Subscribing to the Broker
	command2 = "mosquitto_sub -h '192.168.xxx.xxx' -t 'HumidTempPredict'"
	# process2 = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
	process2 = await asyncio.create_subprocess_shell(command2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
	# Settings and presenting web layout
	web_layout()
	# Reading output and processing it
	total_predictions, samples = [],[]
	counter = 0
	while True:
		output = await process2.stdout.readline()
		await asyncio.sleep(0.5)
		if output:
			counter += 1
			output = output.decode()
			print("Humidity and Temperature read:", output.strip())
			strip_output = output.strip().split(',')
			curren_humidity_val = float(strip_output[0])
			current_temp_celsius = float(strip_output[1])
			current_brightness = float(strip_output[2])
			# The dataset uses farenheit as a feature, hence we need to conver it
			current_temp_farenheit = (current_temp_celsius*1.8)+32
			# print("hum", humidity_val, "tem", temperature_val)
			updated_params = update_parameters(curren_humidity_val, current_temp_farenheit)
			# to_thread is a function in python 3.9 that asynchronously runs a function in a separate thread
			# This is needed to handle the computation time that it takes for the ML to run and predict
			# And the main adventage is that we can continue showing real-time data while running the models
			df_models = await asyncio.to_thread(run_data, updated_params)

			# If it is between dark and very dark, e.g. 10,000 until 30,000 -> weight is prediction is zero. 
			# If it is very bright, e.g. 65,000 -> weight is prediction * 1.325. 
			if current_brightness > 30000:
				bright_weight = ((current_brightness/100000)*0.5)+1
			else:
				bright_weight = 1 # in other words, no weight: this is because anything below 30k is pretty dark...

			final_prediction = df_models[0][0]*bright_weight
			total_predictions.append(final_prediction)
			samples.append(counter)
			write_results(updated_params, current_brightness, bright_weight, current_temp_celsius, df_models[0][0], final_prediction)  # Adding brightness weight
			print("ML prediction:", df_models[0][0], "Brightness Weight:", bright_weight, "Final Prediction:", final_prediction)
			show_ML(df_models[1], total_predictions, samples) # Plots and models
			# Publishing prediction results, for the Pico to read these results
			command3 = f"mosquitto_pub -h '192.168.xxx.xxx' -t 'PredictionResults' -m '{final_prediction}'"
			process3 = await asyncio.create_subprocess_shell(command3, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
		else:
			break
		await asyncio.sleep(0.5)

    # Once the process is stopped, to "clean" the subprocess
	process2.stdout.close()
	return_code = process2.wait()
	if return_code:
		raise subprocess.CalledProcessError(return_code, command2)


async def realtime_data_sidebar():
	command3 = "mosquitto_sub -h '192.168.xxx.xxx' -t 'HumidTempRealTime'"
	process3 = await asyncio.create_subprocess_shell(command3, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
	temps, humids, bright = [],[],[]
	while True:
		output = await process3.stdout.readline()
		output = output.decode()
		strip_output = output.strip().split(',')
		curren_humidity_val = float(strip_output[0])
		current_temperature_val = float(strip_output[1])
		current_brightness = float(strip_output[2])
		temps.append(current_temperature_val)
		humids.append(curren_humidity_val)
		bright.append(current_brightness)
		# Less nicer line charts can be done with the two lines below
		# plot1_title.write("Temperature Real Time (RP Pico WH): " + str(current_temperature_val)) # Celsius
		# plot_slot1.line_chart(temps, color="#843c54", height=150) # use_container_width=False, width=250
		fig = px.line(temps, title="             Temperature Real Time (RP Pico WH)")
		fig.update_traces(line=dict(color='#843c54', width=9))
		fig.update_layout(xaxis_title='Sample Number', yaxis_title='Temperature', height=265)
		plot_slot1.plotly_chart(fig)

		# Less nicer line charts can be done with the two lines below
		# plot2_title.write("Humidity Real Time (RP Pico WH): " + str(curren_humidity_val))
		# plot_slot2.line_chart(humids, color="#66cdaa", height=150)
		fig2 = px.line(humids, title="             Humidity Real Time (RP Pico WH)")
		fig2.update_traces(line=dict(color='#66cdaa', width=9))
		fig2.update_layout(xaxis_title='Sample Number', yaxis_title='Humidity', height=265)
		plot_slot2.plotly_chart(fig2)
		
		plot3_title.write("Brightness Real Time (RP Pico WH): " + str(current_brightness))
		plot_slot3.bar_chart(bright, color="#ffa500", height=140)
		await asyncio.sleep(0.5)

async def main():
	realtime_sidebar = asyncio.create_task(realtime_data_sidebar()) # Task1
	pred_and_display = asyncio.create_task(predict_and_display())   # Task2
	# Return a future aggregating results from the given coroutines/futures (must share the same event loop)
	# Coroutines will be wrapped in a future and scheduled in the event loop.
	await asyncio.gather(pred_and_display, realtime_sidebar)

# inital_caching_params = update_parameters(10,10)
# building_machine_learning_models(inital_caching_params)
asyncio.run(main())
