#!/usr/bin/env python3
"""Program that inserts a new document in a collection based on kwargs"""


def insert_school(mongo_collection, **kwargs):
    """Function that inserts a new document in a collection based on kwargs"""
    mongo_collection.insert(kwargs)
    new_doc = mongo_collection.find(kwargs)
    return new_doc.__dict__['_Cursor__spec']['_id']
