#!/usr/bin/env python3
"""Program that changes all topics of a school document based on the name"""


def update_topics(mongo_collection, name, topics):
    """Function that changes all topics of a school document based on the name"""
    aux = {'$set': {'topics': topics}}
    mongo_collection.update_many({'name': name}, aux)
