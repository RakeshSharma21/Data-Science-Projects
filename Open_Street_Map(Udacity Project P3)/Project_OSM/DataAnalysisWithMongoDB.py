import pprint

def get_db(db_name):
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client[db_name]
    return db

db = get_db('delhiosm')

#function to give unique record for a particular field.
def unique_record(db,fieldName):
    usercol=db.delhiosmcol.distinct(fieldName)
    return len(usercol)

#function to find total number of records in a file.
def size_of_file(db):
    size=db.delhiosmcol.find().count()
    return size

#function to find number of record cound with for a particular field and value.
def record_count(db,fieldName,valueName):
    rec_count=db.delhiosmcol.find({fieldName:valueName}).count()
    return rec_count


#function to create grouping pipeline for a given field and filter values.
def grouping_pipeline(db,groupByField,filterField,filterValue):
    pipeline=[{"$match":{filterField:filterValue}},
        {"$group":{"_id":groupByField,
                        "count":{"$sum":1}}},
             {"$sort":{"count":-1}}]
    return pipeline

#function specifically for a pipeline to count the number of place lie and long and lat co-ordinates after
#rounding them to two decimal places.
def lonlat_pipline():
    pipeline=[ {"$match":{"pos":{"$exists":1}}},
        {"$project":{"lon":{"$arrayElemAt":["$pos",0]},
               "lat":{"$arrayElemAt":["$pos",1]}}},

              {"$project":{"roundlon" : {
              "$subtract":[
                {"$add":['$lon',0.0049999999999999999999999]},
                {"$mod":[{"$add":['$lon',0.0049999999999999999999999]}, 0.01]}
                          ]
                        },
                          "roundlat" : {
            "$subtract":[
                {"$add":['$lat',0.0049999999999999999999999]},
                {"$mod":[{"$add":['$lat',0.0049999999999999999999999]}, 0.01]}
                  ]
                    } }},
           {"$group":{"_id":{"lat":"$roundlon","lon":"$roundlat"}
                        ,"count":{"$sum":1}}},
            {"$sort":{"count":-1}} ,
            {"$limit":5}
             ]
    return pipeline
    


    

def aggregate(db, pipeline):
    return [doc for doc in db.delhiosmcol.aggregate(pipeline)]




#size of the file
print ("size of file "+str(size_of_file(db)))

#no of unique user
uniqueUserCount=unique_record(db,"created.user")
print ("no. of unique users "+str(uniqueUserCount))

#no of nodes count
nodeCount=record_count(db,"type","node")
print ("no. of nodes count "+str(nodeCount))

#no of ways count
waysCount=record_count(db,"type","way")
print ("no. of ways count "+str(waysCount))

#no of fields having amenity
noOfAmenity=record_count(db,"amenity",{"$exists":1})
print ("no. of record with amenity field "+str(noOfAmenity))

#no of amenity with fast food
noOfAmenity=record_count(db,"amenity","fast_food")
print ("no. of record with fast_food amenity "+str(noOfAmenity))

#count of each amenity
pipeline = grouping_pipeline(db,"$amenity","amenity",{"$exists":1})
result = aggregate(db, pipeline)
pprint.pprint(result)

#no of amenties per religion
pipeline = grouping_pipeline(db,"$religion","amenity","place_of_worship")
result = aggregate(db, pipeline)
print ("no. of amenities per religion")
pprint.pprint(result)

#top 5 records which lie in the nearby position
#I take long and lat rounding till 2 decimal places then print the count of no of places fall on particular co-ordinates.
pipeline = lonlat_pipline()
result = aggregate(db, pipeline)
print ("Top five lattitude and longitute position which maximum nearby places.")
pprint.pprint(result) 
