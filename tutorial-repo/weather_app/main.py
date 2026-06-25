import sys
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QPushButton, QInputDialog, QMessageBox)
from get_weather import get_weather  
# ask user for city name and print weather data
if __name__ == "__main__":
    app = QApplication(sys.argv)
    city, ok = QInputDialog.getText(None, 'City Input', 'Enter city name:')
    if ok and city:
        weather = get_weather(city)
        if weather:
            msg = f"Temperature: {weather['temperature']}°C\nHumidity: {weather['humidity']}%\nDescription: {weather['description']}"
        else:
            msg = "City not found."
        QMessageBox.information(None, 'Weather Information', msg)
   
        