import requests

#set value api key from OpenWeatherMap after signup
api_key = '' #"your_api_key_here"
#get weather data from OpenWeatherMap using get request

def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main = data["main"]
        weather = data["weather"][0]
        weather_data = {
            "temperature": main["temp"],
            "humidity": main["humidity"],
            "description": weather["description"]
        }
        return weather_data
    else:
        return None