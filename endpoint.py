import requests
import json

#training_data.to_pandas_dataframe()
#df.columns
#test = df[df['Column1']==0]
#test['digit']
#test.pop('digit')
#js = test.to_json()
#js
#test.values
#test.columns
#js = test.to_json()
#js
#y = json.loads(js)
#y

# URL for the web service, should be similar to:
# 'http://17fb482f-c5d8-4df7-b011-9b33e860c111.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://4de0afd6-0c1e-422a-895e-26f0600405ab.southcentralus.azurecontainer.io/score'

# Two sets of data to score, so we get two results back
data = {"data":
        [
            {'Column1':  0,
             '0': 0.0,
             '1':  0.0,
             '2':  0.0,
             '3':  2.0,
             '4':  13.0,
             '5':  0.0,
             '6':  0.0,
             '7':  0.0,
             '8':  0.0,
             '9':  0.0,
             '10':  0.0,
             '11':  8.0,
             '12':  15.0,
             '13':  0.0,
             '14':  0.0,
             '15': 0.0,
             '16':  0.0,
             '17': 0.0,
             '18': 5.0,
             '19': 16.0,
             '20':  5.0,
             '21': 2.0,
             '22':  0.0,
             '23':  0.0,
             '24': 0.0,
             '25':  0.0,
             '26':  15.0,
             '27':  12.0,
             '28':  1.0,
             '29':  16.0,
             '30':  4.0,
             '31':  0.0,
             '32':  0.0,
             '33':  4.0,
             '34':  16.0,
             '35':  2.0,
             '36':  9.0,
             '37':  16.0,
             '38':  8.0,
             '39':  0.0,
             '40':  0.0,
             '41':  0.0,
             '42':  10.0,
             '43':  14.0,
             '44':  16.0,
             '45':  16.0,
             '46':  4.0,
             '47':  0.0,
             '48':  0.0,
             '49':  0.0,
             '50':  0.0,
             '51':  0.0,
             '52':  13.0,
             '53':  8.0,
             '54':  0.0,
             '55':  0.0,
             '56':  0.0,
             '57':  0.0,
             '58':  0.0,
             '59':  0.0,
             '60':  13.0,
             '61':  6.0,
             '62':  0.0,
             '63':  0.0}
        ]}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
