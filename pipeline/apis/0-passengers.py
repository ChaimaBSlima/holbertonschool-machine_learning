#!/usr/bin/env python3
"""Program that By using the Swapi API, create a method that returns
the list of ships that can hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """Function that By using the Swapi API, create a method that returns
    the list of ships that can hold a given number of passengers"""
    ships_list = []
    url = 'https://swapi-api.hbtn.io/api/starships'
    while url is not None:
        data = requests.get(url).json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            if passengers == 'n/a' or passengers == 'unknown':
                passengers = -1
            if int(passengers) >= passengerCount:
                ships_list.append(ship['name'])
        url = data['next']
    return ships_list
