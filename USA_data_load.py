# coding: utf-8

import os
import zipfile
import cjson
import pandas as pd
import numpy as np
import xmltodict, json
import pickle


from xml.dom.minidom import parse

filenames = os.listdir('USA_data')

raw_text = {}

for zipfilename in filenames:
    with zipfile.ZipFile('USA_data/'+zipfilename) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                try:
                    with z.open(filename) as f:
                        raw_text[filename[:-4]] = json.dumps(xmltodict.parse(f))
                except:
                    print filename
                    pass


usa_list = []

for key, val in raw_text.items():
    usa_dict = {}

    value = cjson.decode(val)
    new_val = value['rootTag']['Award']

    usa_dict['id'] = new_val['AwardID']
    usa_dict['title'] = new_val['AwardTitle']
    usa_dict['objective'] = new_val['AbstractNarration']
    usa_dict['state'] = new_val['Institution']['StateCode']
    
    if key[:2] in ['94', '95', '96', '97']:
        usa_dict['framework_programme'] = 'FP4'
    elif key[:2] in ['98', '99', '00', '01']:
        usa_dict['framework_programme'] = 'FP5'
    elif key[:2] in ['02', '03', '04', '05', '06']:
        usa_dict['framework_programme'] = 'FP6'
    elif key[:2] in ['07', '08', '09', '10', '11', '12', '13']:
        usa_dict['framework_programme'] = 'FP7'
    else:
        usa_dict['framework_programme'] = 'H2020'
    
    try:
        if type(new_val['ProgramReference']) is dict:
            usa_dict['subjects'] = new_val['ProgramElement']['Text'] + new_val['ProgramReference']['Text']
        elif type(new_val['ProgramReference']) is list:
            text = ''
            for i in new_val['ProgramReference']:
                text = text + ' ' + i['Text']

            usa_dict['subjects'] = text
        else:
            pass
    except:
        pass


    try:
        if type(new_val['FoaInformation']) is dict:
            usa_dict['foa'] = new_val['FoaInformation']['Name']
        elif type(new_val['FoaInformation']) is list:
            text = ''
            for i in new_val['FoaInformation']:
                text = text + ' ' + i['Name']

            usa_dict['foa'] = text
        else:
            pass
    except:
        pass

    usa_list.append(usa_dict)

dfUSA = pd.DataFrame(usa_list)
dfUSA.to_csv("usa_data.csv", sep = ';', encoding='utf-8')

