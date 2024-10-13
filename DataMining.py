# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import json

# Configuration options
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 300)

# Open the .json file
with open('data.json') as json_file:
    data = json.load(json_file)

# Function to normalize JSON
def normalize_json(item, level: int, parent_id=None, parent_name=None):
    result = [{
        'id': item.get('id'),
        'RealClass': item.get('name'),
        'Level': level,
        'sift': item.get('sift'),
        'index': item.get('index'),
        'parent_id': parent_id,
        'ParentClass': parent_name
    }]

    if 'children' in item:
        level += 1
        for child in item['children']:
            result.extend(normalize_json(child, level=level, parent_id=item.get('id'), parent_name=item.get('name')))

    return result

# Flatten the JSON
normalized_data = normalize_json(data, level=0)

# Create a DataFrame from the list of dictionaries
df_flat = pd.DataFrame(normalized_data)

# DataFrame information
# Rows and columns (2154 rows and 6 columns)
# print(df_flat.shape)

# Columns -> sift and index can be removed
# columns = df_flat.columns
# print(columns)

# DataFrame columns info/types
# print(df_flat.info())

# Basic transformations
df_flat.drop(df_flat[df_flat['id'] == 'fall11'].index, inplace=True)
df_flat.loc[df_flat['ParentClass'] == 'ImageNet 2011 Fall Release', 'ParentClass'] = 'root_name'
df_flat = df_flat.drop(['sift', 'index'], axis=1)

# Keep only the first name if there is more than one for the class and convert it to lowercase
df_flat['RealClass'] = df_flat['RealClass'].apply(lambda string: string.split(',')[0])
df_flat['RealClass'] = df_flat['RealClass'].str.lower()
df_flat['ParentClass'] = df_flat['ParentClass'].apply(lambda string: string.split(',')[0])
df_flat['ParentClass'] = df_flat['ParentClass'].str.lower()

# print(df_flat.head(30))

# REMOVE DUPLICATES
df_flat = df_flat.drop_duplicates(subset=['RealClass'], keep='first')

print("################################################################################################################")
# Function to establish grandparents

# Initialize the 'GrandparentClass' column
df_flat['GrandparentClass'] = "no_grandparent"
df_flat['grandparent_id'] = 'no_id'

# Iterate over each row of the DataFrame
for idx, row in df_flat.iterrows():
    # Get the current parent's id
    parent_id = row['parent_id']
    # print(f'The parent id \n {parent_id}')

    if pd.notna(parent_id):  # Check that it is not NaN
        # Find the grandparent ID
        grandparent_id = df_flat.loc[df_flat['id'] == parent_id, 'parent_id'].values
        grandparent_name = df_flat.loc[df_flat['id'] == parent_id, 'ParentClass'].values
        # print(f'The grandparent id \n {grandparent_id} and {grandparent_name}')

        if grandparent_name.size > 0:
            df_flat.at[idx, 'GrandparentClass'] = grandparent_name
            df_flat.at[idx, 'grandparent_id'] = grandparent_id

print("################################################################################################################")

# New order of columns
new_order = ['Level', 'id', 'RealClass', 'parent_id', 'ParentClass', 'grandparent_id', 'GrandparentClass']
df_flat = df_flat[new_order]

# Function to convert to .csv
#df_flat.to_csv('Class_Hierarchy.csv', index=False)

# print(df_flat.head(10))

# Duplicate the df but now instead of RealClass use PredictedPrimeClass (for the join)
# ParentClass -> PredictedParentClass, GrandparentClass -> PredictedGrandparentClass

# Dictionary to rename the columns
new_names = {
    'RealClass': 'PredPrimeClass',
    'ParentClass': 'PredParentClass',
    'GrandparentClass': 'PredGrandparentClass'
}

# Rename the DataFrame columns
df_renamed = df_flat.rename(columns=new_names)
df_renamed = df_renamed.drop(['Level', 'id', 'parent_id', 'grandparent_id'], axis=1)

# Function to convert to .csv
#df_renamed.to_csv('Predicted_Class_Hierarchy.csv', index=False)

# print(df_renamed.head(15))

print("################################################################################################################")

print(df_flat.sample(10))
