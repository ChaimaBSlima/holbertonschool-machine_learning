#!/usr/bin/env python3
"""Program that By using the Swapi API, create a method that
returns the list of names of the home planets of all sentient species"""
import json
import requests


def sentientPlanets():
    """Function that By using the Swapi API, create a method that returns
    the list of ships that can hold a given number of passengers"""
    planet_list = []
    url = 'https://swapi-api.hbtn.io/api/species'
    while url is not None:
        data = requests.get(url).json()
        for species in data['results']:
            if ((species['designation'] == 'sentient'
                 or species['designation'] == 'reptilian')):
                if species['homeworld'] is not None:
                    hw = requests.get(species['homeworld']).json()
                    planet_list.append(hw['name'])
        url = data['next']
    return planet_list
