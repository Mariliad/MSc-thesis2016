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

usa_all = []

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

for key, val in raw_text.items():
    usa_dict = {}

    value = cjson.decode(val)
    new_val = value['rootTag']['Award']

    # usa_dict['id'] = new_val['AwardID']
    # usa_dict['title'] = new_val['AwardTitle']
    # usa_dict['objective'] = new_val['AbstractNarration']
    try:
        usa_dict['merged'] = new_val['AwardTitle'] + new_val['AbstractNarration']
    except:
        pass

    if key[:2] in ['94', '95', '96', '97', '98', '99']:
    	usa_dict['year'] = int('19' + key[:2])
    else:
    	usa_dict['year'] = int('20' + key[:2])
    usa_all.append(usa_dict)


df_usa = pd.DataFrame(usa_all)
df_usa = df_usa.dropna(how='any')

print df_usa.columns
print df_usa.shape
print df_usa.head(2)



df_usa.to_pickle('usa_gensim/pickle_data/dfUSA')
