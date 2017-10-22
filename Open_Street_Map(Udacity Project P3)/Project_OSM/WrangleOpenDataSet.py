import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import csv
import xlrd
import codecs
import json

#Name of Osm file used for the analysis
OSMFILE = "new-delhi_india_sample.osm"

#Regular expression to find out the invalid street name
street_type_re =  re.compile(r'[\+/&<>;\'"\?%#$@\\.]')

#Regular expression to find out street names ending with the postal code
street_with_postal=re.compile(r'\d{6}$')

#Regular expression to find the valid postal codes.
post_code_re=re.compile(r'\d{6}$')

valid_postcode=set()

#valid postal code file
datafile="ListofPinCodesofDelhi.xls"

#list of created dictionary
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#mapping dictionary for problematic data
mapping = { "St": "Street",
            "St.": "Street",
            "Ave" : "Avenue",
            "Ave." : "Avenue",
            "Rd" : "Road",
            "Rd." : "Road",
            "Blvd" : "Boulevard",
            "Blvd." : "Boulevard",
            "Cir" : "Circle",
            "Cir." : "Circle",
            "Ct" : "Court",
            "Ct." : "Court",
            "Dr" : "Drive",
            "Dr." : "Drive",
            "Pl" : "Place",
            "Pl." : "Place",
            "h/no":"",
            "Vill.":"Village",
            "wz-10":"",
            "B-3/27":"",
            "P.O.":"Post Office"
            }




#function to load the valid postcode master data
def load_pin_code_master_data(datafile):   
    workbook = xlrd.open_workbook(datafile)
    sheet = workbook.sheet_by_index(0)
    for r in range(sheet.nrows-1):
        valid_postcode.add(sheet.cell_value(r+1,2))
    return valid_postcode


                    
#function to update the problematic postcode with the valid ones.
def update_post_code(postcode):
    if post_code_re.search(str(postcode)):
        valid_postcode=load_pin_code_master_data("ListofPinCodesofDelhi.xls")
        if float(postcode[-6:]) in valid_postcode:
            return postcode[-6:]

        
#function to update the problematic street name with the valid ones.
def update_street_name(name, mapping):  
    ll=[]
    updName=""
    if street_type_re.search(name):   
        for k,v in mapping.iteritems():
            if k in name and "," not in name:
                updName= name.replace(k,v)
                if street_with_post_code(updName):
                    updName=street_with_post_code(name)               
            elif k in name and "," in name:
                name1= name.replace(k,v)
                ll=name1.split(",")
                updName=ll[1]
                if street_with_post_code(updName):
                    updName=street_with_post_code(name)    
    elif "," in name:
        ll=name.split(",")
        updName= ll[1]
        if street_with_post_code(updName):
            updName=street_with_post_code(name)  
    elif street_with_post_code(name):
        updName=street_with_post_code(name)
        
    return updName

#function to clean the street name ending with postcode and returning the street name only.
def street_with_post_code(name):
    if street_with_postal.search(name):
        return street_with_postal.sub("",name)

    
#to check whether the given value is a number or not.
def is_number(field_val):
    try:
        float(field_val)
        return True
    except ValueError:
        pass
    
    try:
        import unicodedata as un
        un.numeric(field_val)
        return True
    except(TypeError,ValueError):
        pass

    return False


#function to load the creation dictionary with the valid values.
def load_node_creation_field(element):
    localDic={}
    for attr in element.attrib:
        if attr in CREATED:
            localDic[attr]=element.get(attr)
    return localDic

#function to load the address field after cleaning the street and postcode.
def load_address_field(valfork,valforv):
    ll=[]
    valueforv=""
    if valfork.startswith("addr:"):
        ll=valfork.split(":")
        if len(ll)<3:
            if valfork=="addr:street":
                if update_street_name(valforv, mapping):                       
                    valueforv=update_street_name(valforv, mapping)
                else:
                    valueforv=valforv
            elif valfork=="addr:postcode":
                if update_post_code(valforv):
                    valueforv=update_post_code(valforv)
                    
            else:
                valueforv=valforv
        
    return valueforv


#finally shaping the elements to convert xml file to json.
def shape_element(element):
    node = {}
    createddic={}
    addrdic={}
    longlat=[]
    reflist=[]
    
    #including only node and way tag for analysis.
    if element.tag == "node" or element.tag == "way" :
        #creating longlat entry in desired format.
        if is_number(str(element.get('lat'))) and is_number(str(element.get('lon'))):               
            longlat=[float(element.get('lat')),float(element.get('lon'))]
        
        #shaping creation records
        createddic=load_node_creation_field(element)
        
        #dealing with the address field
        for elem in element.iter("tag"):
            valfork=elem.get("k")
            valforv=elem.get("v")
            if load_address_field(valfork,valforv):
                #print load_address_field(valfork,valforv)
                addrdic[valfork[5:]]=load_address_field(valfork,valforv)                                             
            elif valfork.startswith("name:"):
                ll=valfork.split(":")
                if ll[1]=="en":
                    node[ll[1]]=valforv                    
            else:
                if ":" not in valfork:
                    node[valfork]=valforv
                
        #adding ref records of way tag
        if element.tag=="way":
            for elem1 in element.iter("nd"):
                valforref=elem1.get("ref")
                reflist.append(valforref)
                
        node["type"]=element.tag
        node["id"]=element.get("id")
        node["visible"]=element.get("visible")
        
        #finally adding all the records in desired shape.
        if bool(addrdic):
            node["address"]=addrdic
        if len(longlat)>0:
            node["pos"]=longlat
        if len(reflist)>0:
            node["node_refs"]=reflist
        node["created"]=createddic
        
        return node
    else:
        return None

# converting shaped element to json file.
def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def main():
    data = process_map(OSMFILE, True)

if __name__ == "__main__":
    main()
