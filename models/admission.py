"""Document description
GRUD for admission collection
"""

# Author: Rui Li <li.8950@osu.edu>
# License: MIT
# Copyright: Rui Li
# Date: 9/17/19

import json
from bson.objectid import ObjectId
from common.database import Database

class JSONEncoder(json.JSONEncoder):
    '''
    database object decoder, id
    https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    '''
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

class Admission(object):
    @classmethod
    def find_all(cls):
        '''
        find all patient record
        :return: all patient record in json format
        '''
        data = list(Database.findall("admission"))
        data = JSONEncoder().encode(data)
        if data is not None:
            return data

    @classmethod
    def find_all_countlg2(cls):
        '''
        find all patient that has more than two visit
        :return: patient record in json format
        @_id SUBJECT_ID: patient id
        @count: visit count
        @data: visit history, detailed data
        '''
        query = [{
            '$group': {
                '_id': {'SUBJECT_ID': '$SUBJECT_ID'},
                'count': {'$sum': 1},
                'data': {'$addToSet': '$$ROOT'}}
        }, {
            '$match': {
                'count': {'$gt': 1}}
        }]
        data = list(Database.have_count("admission", query))
        data = JSONEncoder().encode(data)
        if data is not None:
            return data
