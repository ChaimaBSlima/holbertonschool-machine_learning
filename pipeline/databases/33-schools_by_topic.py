#!/usr/bin/env python3
"""Program that returns the list of school having a specific topic"""


def schools_by_topic(mongo_collection, topic):
    """Function that returns the list of school having a specific topic"""
    all_items = mongo_collection.find()
    docs = []
    doc_filter = []
    for elem in all_items:
        docs.append(elem)
    for elem in docs:
        if 'topics' in elem.keys():
            if topic in elem['topics']:
                doc_filter.append(elem)
    return doc_filter
