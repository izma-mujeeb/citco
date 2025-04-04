import pandas as pd 

csv_file = './uploads/nserc.csv' 
df = pd.read_csv(csv_file, usecols=['Name', 'Amount($)', 'Program']) # read data from csv file 

df_cleaned = (df[df['Program'] == 'Discovery Grants Program - Individual']).drop_duplicates(subset='Name') # extract researchers who recieved a DG 

json_data = df_cleaned.to_json(orient='records', lines=True) # covert data to a json file 

with open('output.json', 'w') as json_file:
    json_file.write(json_data) # write data to a json file 