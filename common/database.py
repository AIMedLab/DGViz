"""database config and abstraction
"""

# Author: Rui Li <li.8950@osu.edu>
# License: MIT
# Copyright: Rui Li
# Date: 9/16/19

from pymongo import MongoClient # Database connector
import os

class Database(object):

    #database configuration
    mongodb_host = os.environ.get('MONGO_HOST', 'localhost')
    mongodb_port = int(os.environ.get('MONGO_PORT', '27017'))

    DATABASE = None

    @staticmethod
    def initialize():
        client = MongoClient(Database.mongodb_host, Database.mongodb_port)
        Database.DATABASE = client.ehrdb

    @staticmethod
    def insert(collection, data):
        Database.DATABASE[collection].insert(data)

    @staticmethod
    def findall(collection):
        return Database.DATABASE[collection].find({})

    @staticmethod
    def find(collection, query):
        return Database.DATABASE[collection].find(query)

    @staticmethod
    def find_one(collection, query):
        return Database.DATABASE[collection].find_one(query)


    @staticmethod
    def have_count(collection, query):
        # return aggregation results
        return Database.DATABASE[collection].aggregate(query)




