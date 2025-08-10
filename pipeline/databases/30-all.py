#!/usr/bin/env python3
"""Program that lists all documents in a collection"""


def list_all(mongo_collection):
    """Function that lists all documents in a collection"""
    docs = []
    list_all = mongo_collection.find()
    for i in list_all:
        docs.append(i)
    return docs
