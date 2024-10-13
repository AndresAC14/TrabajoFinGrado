# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# Opciones de configuración
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 400)

# Configuración global del tamaño de la fuente
plt.rcParams.update({'font.size': 18})


# Importar el dataframe
df = pd.read_csv('Training_Set.csv')
hierarchy = pd.read_csv('Class_Hierarchy.csv')
predictedHierarchy = pd.read_csv('Predicted_Class_Hierarchy.csv')

# Remove the first column and the 'r' column because they are not needed
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(df.columns[12], axis=1, inplace=True)

# Basic exploratory analysis plots

# Most common RealClass plot
sns.barplot(df['RealClass'].value_counts().head(10))
plt.title('Most Common Real Class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

# Most common PredClass plot
sns.barplot(df['PredClass'].value_counts().head(10))
plt.title('Most Common Predicted Class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

# Most common PredPrimeClass plot
sns.barplot(df['PredPrimeClass'].value_counts().head(10))
plt.title('Most Common Predicted Prime Class')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

print(df.sample(10))

print("################################################################################################################")
# JOIN the large dataframe with the hierarchy dataframe on 'RealClass'
df['RealClass'] = df['RealClass'].str.lower()
df['PredClass'] = df['PredClass'].str.lower()
df['PredPrimeClass'] = df['PredPrimeClass'].str.lower()
df = pd.merge(df, hierarchy, on='RealClass', how='left')
df['Level'] = df['Level'].astype('Int64')

# Merge the Predicted hierarchies
df = pd.merge(df, predictedHierarchy, on='PredPrimeClass', how='left')

# Function for Real Parent and Grandparent accuracies
# Add accuracy columns
df['Real_Accuracies'] = (df['RealClass'] == df['PredPrimeClass']).astype(int)
df['Parent_Accuracies'] = (df['ParentClass'] == df['PredParentClass']).astype(int)
df['Grandparent_Accuracies'] = (df['GrandparentClass'] == df['PredGrandparentClass']).astype(int)

# Logical condition for accuracy 2
# Create the total accuracies column
df['Total_Accuracies'] = ((df['RealClass'] != df['PredPrimeClass']) &
                           ((df['ParentClass'] == df['PredParentClass']) |
                            (df['GrandparentClass'] == df['PredGrandparentClass']) |
                            (df['ParentClass'] == df['PredGrandparentClass']) |
                            (df['GrandparentClass'] == df['PredParentClass']))).astype(int)

# print(df.head(15))

print(df.sample(10))

'''
# Example: show all the Persian cats
persian = df[df['RealClass'] == 'persian cat']
print(persian)
'''

print("################################################################################################################")

print('Total accuracy percentage (All algorithms)')
# REAL CLASS REGARDING PREDICTED PRIME CLASS since it is the one involved in image improvement

# Filter rows where RealClass and PredPrimeClass are equal
matches = df[df['RealClass'] == df['PredPrimeClass']]
# print(matches)

# Group the equal ones and count them
df_1 = matches.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Accuracies')
# print('df1', df_1.head(2))

df_2 = df.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Total')
# print('df2', df_2.head(2))

df_classes = pd.merge(df_1, df_2, on='RealClass', how='left')

# Create the result column
df_classes['Percentage'] = ((df_classes['Accuracies'] / df_classes['Total']) * 100).__round__(3)
df_classes = df_classes.sort_values(by='Percentage', ascending=False)
print(df_classes.head(5))

# Now create a graph that shows RealClass vs Percentage
sns.barplot(data=df_classes.head(10), x='RealClass', y='Percentage')
plt.title('Top 10 Most Predicted Classes (GLOBAL)')
plt.xlabel('Class')
plt.xticks(rotation=45, ha='right')
plt.show()

print("################################################################################################################")
# From this point, advanced exploratory analysis begins

# Improvement algorithms we handle:
#  'Histogram Equalization (HE)'
#  'Adaptive Histogram Equalization Limited by Contrast (AHELC)'
#  'Gamma Correction (GC)'
#  'Log Transformation (LT)'
print("################################################################################################################")

# Accuracies for Histogram Equalization algorithm with PredPrimeClass
df_HE = matches[matches['Algorithm'] == 'Histogram Equalization']
df_HE = df_HE.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Accuracies')

# Represents how important each class is regarding this algorithm
df_HE['Percentage'] = ((df_HE['Accuracies'] / df_HE['Accuracies'].sum()) * 100).__round__(3)
df_HE = df_HE.sort_values(by='Accuracies', ascending=False)
accuracies_HE = df_HE['Accuracies'].sum()

# Show the resulting DataFrame
print(df_HE.head(5))
print('Total Accuracies HE', accuracies_HE)

# TOP 10 with HE
sns.barplot(data=df_HE.head(10), x='RealClass', y='Accuracies')
plt.title('Top 10 Most Predicted Classes with HE')
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.xticks(rotation=45, ha='right')
plt.show()

print("################################################################################################################")
print("################################################################################################################")

# Accuracies for Adaptive Histogram Equalization Limited by Contrast algorithm
df_AHELC = matches[matches['Algorithm'] == 'Adaptive Histogram Equalization Limited by Contrast']
df_AHELC = df_AHELC.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Accuracies')

# Represents how important each class is regarding this algorithm
df_AHELC['Percentage'] = ((df_AHELC['Accuracies'] / df_AHELC['Accuracies'].sum()) * 100).__round__(3)
df_AHELC = df_AHELC.sort_values(by='Accuracies', ascending=False)
accuracies_AHELC = df_AHELC['Accuracies'].sum()

# Show the resulting DataFrame
print(df_AHELC.head(5))
print('Total Accuracies AHELC', accuracies_AHELC)

# TOP 10 with AHELC
sns.barplot(data=df_AHELC.head(10), x='RealClass', y='Accuracies')
plt.title('Top 10 Most Predicted Classes with AHELC')
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.xticks(rotation=45, ha='right')
plt.show()

print("################################################################################################################")
print("################################################################################################################")

# 'Gamma Correction (GC)'
df_GC = matches[matches['Algorithm'] == 'Gamma Correction']
df_GC = df_GC.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Accuracies')

# Represents how important each class is regarding this algorithm
df_GC['Percentage'] = ((df_GC['Accuracies'] / df_GC['Accuracies'].sum()) * 100).__round__(3)
df_GC = df_GC.sort_values(by='Accuracies', ascending=False)
accuracies_GC = df_GC['Accuracies'].sum()

# Show the resulting DataFrame
print(df_GC.head(5))
print('Total Accuracies GC', accuracies_GC)

# TOP 10 with GC
sns.barplot(data=df_GC.head(10), x='RealClass', y='Accuracies')
plt.title('Top 10 Most Predicted Classes with GC')
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.xticks(rotation=45, ha='right')
plt.show()

print("################################################################################################################")
print("################################################################################################################")

# 'Log Transformation (LT)'
df_LT = matches[matches['Algorithm'] == 'Log Transformation']
df_LT = df_LT.groupby('RealClass')['PredPrimeClass'].count().reset_index(name='Accuracies')

# Represents how important each class is regarding this algorithm
df_LT['Percentage'] = ((df_LT['Accuracies'] / df_LT['Accuracies'].sum()) * 100).__round__(3)
df_LT = df_LT.sort_values(by='Accuracies', ascending=False)
accuracies_LT = df_LT['Accuracies'].sum()

# Show the resulting DataFrame
print(df_LT.head(5))
print('Total Accuracies LT', accuracies_LT)

# TOP 10 with LT
sns.barplot(data=df_LT.head(10), x='RealClass', y='Accuracies')
plt.title('Top 10 Most Predicted Classes with LT')
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.xticks(rotation=45, ha='right')
plt.show()

print("################################################################################################################")
print("################################################################################################################")

# Algorithm with the most accuracies
accuracy_columns = ['Histogram Equalization', 'Adaptive Histogram Equalization Limited by Contrast', 'Gamma Correction', 'Log Transformation']
accuracy_values = [accuracies_HE, accuracies_AHELC, accuracies_GC, accuracies_LT]
df_accuracies = pd.DataFrame([accuracy_values], columns=accuracy_columns, index=['Accuracies'])

print(df_accuracies)

# Pie chart showing the percentage of accuracies by algorithm
plt.figure(figsize=(8, 8))
plt.pie(df_accuracies.iloc[0].values, labels=df_accuracies.columns, autopct='%1.1f%%', startangle=90)
plt.title('Accuracies by Algorithm')
plt.show()

print("################################################################################################################")
print("################################################################################################################")

print('Histogram Equalization (HE)')
# See how much the image/class improves with Histogram Equalization
improvement_HE = df[df['Algorithm'] == 'Histogram Equalization']
improvement_HE = improvement_HE.drop(['MeanPrime', 'Median', 'StdDev', 'MedianPrime', 'StdDevPrime', 'Algorithm', 'PredClass'], axis=1)

# Percentage improvement in probability of being correctly predicted = ((new - old) / old) * 100 )
improvement_HE['% Improvement'] = (((improvement_HE['PredPrimeClassProb'] - improvement_HE['PredClassProb']) / improvement_HE['PredClassProb']) * 100).__round__(3)
print(improvement_HE.head(10))

# Which class improves the most?

# Calculate the mean of all elements that have the same class
class_HE = improvement_HE.groupby('RealClass')['% Improvement'].mean().__round__(3).reset_index(name='% Improvement Per Class')
class_HE = class_HE.sort_values(by='% Improvement Per Class', ascending=False)
# print(class_HE.head(10))

# Top 10 classes most improved with HE
sns.barplot(data=class_HE.head(10), x='RealClass', y='% Improvement Per Class')
plt.xlabel('Real Class')
plt.ylabel('% Average Improvement Per Class')
plt.title('Top 10 Most Improved Classes with HE')
# Rotate the labels
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

print("################################################################################################################")
print("################################################################################################################")

print('Adaptive Histogram Equalization Limited by Contrast (AHELC)')
# See how much the image/class improves with Adaptive Histogram Equalization Limited by Contrast
improvement_AHELC = df[df['Algorithm'] == 'Adaptive Histogram Equalization Limited by Contrast']
improvement_AHELC = improvement_AHELC.drop(['MeanPrime', 'Median', 'StdDev', 'MedianPrime', 'StdDevPrime', 'Algorithm', 'PredClass'], axis=1)

# Percentage improvement = ((new - old) / old) * 100 )
improvement_AHELC['% Improvement'] = (((improvement_AHELC['PredPrimeClassProb'] - improvement_AHELC['PredClassProb']) / improvement_AHELC['PredClassProb']) * 100).__round__(3)
print(improvement_AHELC.head(10))

# Which class improves the most?

# Calculate the mean of all elements that have the same class
class_AHELC = improvement_AHELC.groupby('RealClass')['% Improvement'].mean().__round__(3).reset_index(name='% Improvement Per Class')
class_AHELC = class_AHELC.sort_values(by='% Improvement Per Class', ascending=False)
print('Top 10 Best Classes')
print(class_AHELC.head(10))

# Top 10 classes most improved with AHELC
sns.barplot(data=class_AHELC.head(10), x='RealClass', y='% Improvement Per Class')
plt.xlabel('Real Class')
plt.ylabel('% Average Improvement Per Class')
plt.title('Top 10 Most Improved Classes with AHELC')
# Rotate the labels
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

print("################################################################################################################")
print("################################################################################################################")

print('Gamma Correction (GC)')
# See how much the image/class improves with Gamma Correction
improvement_GC = df[df['Algorithm'] == 'Gamma Correction']
improvement_GC = improvement_GC.drop(['MeanPrime', 'Median', 'StdDev', 'MedianPrime', 'StdDevPrime', 'Algorithm', 'PredClass'], axis=1)

# Percentage improvement = ((new - old) / old) * 100 )
improvement_GC['% Improvement'] = (((improvement_GC['PredPrimeClassProb'] - improvement_GC['PredClassProb']) / improvement_GC['PredClassProb']) * 100).__round__(3)
print(improvement_GC.head(10))

# Which class improves the most?

# Calculate the mean of all elements that have the same class
class_GC = improvement_GC.groupby('RealClass')['% Improvement'].mean().__round__(3).reset_index(name='% Improvement Per Class')
class_GC = class_GC.sort_values(by='% Improvement Per Class', ascending=False)
print('Top 10 Best Classes')
print(class_GC.head(10))

# Top 10 classes most improved with GC
sns.barplot(data=class_GC.head(10), x='RealClass', y='% Improvement Per Class')
plt.xlabel('Real Class')
plt.ylabel('% Average Improvement Per Class')
plt.title('Top 10 Most Improved Classes with GC')
# Rotate the labels
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

print("################################################################################################################")
print("################################################################################################################")

print('Log Transformation (LT)')
# See how much the image/class improves with Log Transformation
improvement_LT = df[df['Algorithm'] == 'Log Transformation']
improvement_LT = improvement_LT.drop(['MeanPrime', 'Median', 'StdDev', 'MedianPrime', 'StdDevPrime', 'Algorithm', 'PredClass'], axis=1)

# Percentage improvement = ((new - old) / old) * 100 )
improvement_LT['% Improvement'] = (((improvement_LT['PredPrimeClassProb'] - improvement_LT['PredClassProb']) / improvement_LT['PredClassProb']) * 100).__round__(3)
print(improvement_LT.head(10))

# Which class improves the most?

# Calculate the mean of all elements that have the same class
class_LT = improvement_LT.groupby('RealClass')['% Improvement'].mean().__round__(3).reset_index(name='% Improvement Per Class')
class_LT = class_LT.sort_values(by='% Improvement Per Class', ascending=False)
print('Top 10 Best Classes')
print(class_LT.head(10))

# Top 10 classes most improved with LT
sns.barplot(data=class_LT.head(10), x='RealClass', y='% Improvement Per Class')
plt.xlabel('Real Class')
plt.ylabel('% Average Improvement Per Class')
plt.title('Top 10 Most Improved Classes with LT')
# Rotate the labels
plt.xticks(rotation=45, ha='right')
# Show the plot
plt.show()

print("################################################################################################################")
print("################################################################################################################")

# Number of classes per level
levels = df['Level'].value_counts().reset_index('Level')
levels = levels.sort_values(by='Level', ascending=True)

print(levels)

print("################################################################################################################")
print("################################################################################################################")

# Comparison of Improvement by Level with Image Improvement Algorithms
def compare_levels(algorithm):

    level_improvement = []
    df_improvement = pd.DataFrame
    top_classes_level = pd.DataFrame(columns=['Level', 'RealClass', '% Improvement Per Class'])

    if algorithm == 'HE':
        df_improvement = improvement_HE

    elif algorithm == 'AHELC':
        df_improvement = improvement_AHELC

    elif algorithm == 'LT':
        df_improvement = improvement_LT

    elif algorithm == 'GC':
        df_improvement = improvement_GC

    for i in levels['Level'].values:
        nodes = df_improvement[df_improvement['Level'] == i]
        improvement = nodes['% Improvement'].mean().__round__(2)
        level_improvement.append((i, improvement))

        classes = nodes.groupby('RealClass')['% Improvement'].mean().__round__(2).reset_index(name='% Improvement Per Class')
        classes = classes.sort_values(by='% Improvement Per Class', ascending=False).head(5)

        # Add the level to the classes for the final DataFrame
        classes['Level'] = i

        # Append to the top_classes_level DataFrame
        top_classes_level = pd.concat([top_classes_level, classes])

        # Plot showing the top per level
        sns.barplot(data=top_classes_level[top_classes_level['Level'] == i], x='RealClass', y='% Improvement Per Class')
        plt.xlabel('Real Class')
        plt.ylabel('% Average Improvement Per Class')
        plt.title(f'Top Classes with {algorithm} at Level {i}')
        # Rotate the labels
        plt.xticks(rotation=45, ha='right')
        # Show the plot
        plt.show()

    level_improvement = pd.DataFrame(level_improvement, columns=['Level', 'Improvement'])

    # Plot comparing improvement by levels, i.e., the 13 levels and the improvement in each of them
    plt.figure(figsize=(10, 10))
    plt.plot(level_improvement['Level'], level_improvement['Improvement'], marker='o', linestyle='-', color='b')
    plt.title(f'Improvement by Level with {algorithm}')
    plt.xlabel('Level')
    plt.ylabel('% Improvement')
    plt.xticks(levels['Level'].values)
    plt.grid(True)
    plt.show()

    return level_improvement


level_improvement_HE = compare_levels('HE')
level_improvement_AHELC = compare_levels('AHELC')
level_improvement_GC = compare_levels('GC')
level_improvement_LT = compare_levels('LT')

print("################################################################################################################")
print("################################################################################################################")


def accuracy_hierarchy(algorithm, level):

    df_result = pd.DataFrame

    if algorithm == 'HE':
        df_result = improvement_HE

    elif algorithm == 'AHELC':
        df_result = improvement_AHELC

    elif algorithm == 'LT':
        df_result = improvement_LT

    elif algorithm == 'GC':
        df_result = improvement_GC

    lv = level.split('_')[1]

    # Group by class
    df_result = df_result.groupby('RealClass')[level].sum().reset_index()
    df_result = df_result.sort_values(by=level, ascending=False)

    sns.barplot(data=df_result.head(10), x='RealClass', y=level)
    plt.xlabel('Class')
    plt.ylabel('Accuracies')
    plt.title(f'Top 10 Predicted Classes at Level {lv} with {algorithm}')
    # Rotate the labels
    plt.xticks(rotation=45, ha='right')
    # Show the plot
    plt.show()

    return df_result


print("################################################################################################################")
# Histogram Equalization

accuracyHE1 = accuracy_hierarchy('HE', 'Accuracies_Real')
accuracyHE2 = accuracy_hierarchy('HE', 'Accuracies_Parent')
accuracyHE3 = accuracy_hierarchy('HE', 'Accuracies_Grandparent')
accuracyHE4 = accuracy_hierarchy('HE', 'Accuracies_Total')

family_HE = pd.merge(accuracyHE1, accuracyHE2, on='RealClass', how='left')
family_HE = pd.merge(family_HE, accuracyHE3, on='RealClass', how='left')

# FUNCTION FOR STACKED BAR CHART
# Filter to show the first 10 classes
family_HE = family_HE.head(10)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Configure positions and width of the bars
bar_width = 0.5
r = range(len(family_HE['RealClass']))

# Stack the columns
p1 = plt.bar(r, family_HE['Accuracies_Real'], color='b', edgecolor='white', width=bar_width, label='Accuracies_Real')
p2 = plt.bar(r, family_HE['Accuracies_Parent'], bottom=family_HE['Accuracies_Real'], color='r', edgecolor='white', width=bar_width, label='Accuracies_Parent')
p3 = plt.bar(r, family_HE['Accuracies_Grandparent'], bottom=family_HE['Accuracies_Real'] + family_HE['Accuracies_Parent'], color='g', edgecolor='white', width=bar_width, label='Accuracies_Grandparent')

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.title('Variability of Accuracies among Real, Parent, and Grandparent with HE')
plt.xticks(r, family_HE['RealClass'], rotation=45, ha='right')
plt.legend()
plt.show()

print("################################################################################################################")
# Adaptive Histogram Equalization Limited by Contrast
accuracyAHELC1 = accuracy_hierarchy('AHELC', 'Accuracies_Real')
accuracyAHELC2 = accuracy_hierarchy('AHELC', 'Accuracies_Parent')
accuracyAHELC3 = accuracy_hierarchy('AHELC', 'Accuracies_Grandparent')
accuracyAHELC4 = accuracy_hierarchy('AHELC', 'Accuracies_Total')

family_AHELC = pd.merge(accuracyAHELC1, accuracyAHELC2, on='RealClass', how='left')
family_AHELC = pd.merge(family_AHELC, accuracyAHELC3, on='RealClass', how='left')

# FUNCTION FOR STACKED BAR CHART
# Filter to show the first 10 classes
family_AHELC = family_AHELC.head(10)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Configure positions and width of the bars
bar_width = 0.5
r = range(len(family_AHELC['RealClass']))

# Stack the columns
p1 = plt.bar(r, family_AHELC['Accuracies_Real'], color='b', edgecolor='white', width=bar_width, label='Accuracies_Real')
p2 = plt.bar(r, family_AHELC['Accuracies_Parent'], bottom=family_AHELC['Accuracies_Real'], color='r', edgecolor='white', width=bar_width, label='Accuracies_Parent')
p3 = plt.bar(r, family_AHELC['Accuracies_Grandparent'], bottom=family_AHELC['Accuracies_Real'] + family_AHELC['Accuracies_Parent'], color='g', edgecolor='white', width=bar_width, label='Accuracies_Grandparent')

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.title('Variability of Accuracies among Real, Parent, and Grandparent with AHELC')
plt.xticks(r, family_AHELC['RealClass'], rotation=45, ha='right')
plt.legend()
plt.show()

print("################################################################################################################")
# Gamma Correction
accuracyGC1 = accuracy_hierarchy('GC', 'Accuracies_Real')
accuracyGC2 = accuracy_hierarchy('GC', 'Accuracies_Parent')
accuracyGC3 = accuracy_hierarchy('GC', 'Accuracies_Grandparent')
accuracyGC4 = accuracy_hierarchy('GC', 'Accuracies_Total')

family_GC = pd.merge(accuracyGC1, accuracyGC2, on='RealClass', how='left')
family_GC = pd.merge(family_GC, accuracyGC3, on='RealClass', how='left')

# FUNCTION FOR STACKED BAR CHART
# Filter to show the first 10 classes
family_GC = family_GC.head(10)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Configure positions and width of the bars
bar_width = 0.5
r = range(len(family_GC['RealClass']))

# Stack the columns
p1 = plt.bar(r, family_GC['Accuracies_Real'], color='b', edgecolor='white', width=bar_width, label='Accuracies_Real')
p2 = plt.bar(r, family_GC['Accuracies_Parent'], bottom=family_GC['Accuracies_Real'], color='r', edgecolor='white', width=bar_width, label='Accuracies_Parent')
p3 = plt.bar(r, family_GC['Accuracies_Grandparent'], bottom=family_GC['Accuracies_Real'] + family_GC['Accuracies_Parent'], color='g', edgecolor='white', width=bar_width, label='Accuracies_Grandparent')

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.title('Variability of Accuracies among Real, Parent, and Grandparent with GC')
plt.xticks(r, family_GC['RealClass'], rotation=45, ha='right')
plt.legend()
plt.show()

print("################################################################################################################")
# Log Transformation
accuracyLT1 = accuracy_hierarchy('LT', 'Accuracies_Real')
accuracyLT2 = accuracy_hierarchy('LT', 'Accuracies_Parent')
accuracyLT3 = accuracy_hierarchy('LT', 'Accuracies_Grandparent')
accuracyLT4 = accuracy_hierarchy('LT', 'Accuracies_Total')

family_LT = pd.merge(accuracyLT1, accuracyLT2, on='RealClass', how='left')
family_LT = pd.merge(family_LT, accuracyLT3, on='RealClass', how='left')

# FUNCTION FOR STACKED BAR CHART
# Filter to show the first 10 classes
family_LT = family_LT.head(10)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Configure positions and width of the bars
bar_width = 0.5
r = range(len(family_LT['RealClass']))

# Stack the columns
p1 = plt.bar(r, family_LT['Accuracies_Real'], color='b', edgecolor='white', width=bar_width, label='Accuracies_Real')
p2 = plt.bar(r, family_LT['Accuracies_Parent'], bottom=family_LT['Accuracies_Real'], color='r', edgecolor='white', width=bar_width, label='Accuracies_Parent')
p3 = plt.bar(r, family_LT['Accuracies_Grandparent'], bottom=family_LT['Accuracies_Real'] + family_LT['Accuracies_Parent'], color='g', edgecolor='white', width=bar_width, label='Accuracies_Grandparent')

# Add labels and title
plt.xlabel('Class')
plt.ylabel('Accuracies')
plt.title('Variability of Accuracies among Real, Parent, and Grandparent with LT')
plt.xticks(r, family_LT['RealClass'], rotation=45, ha='right')
plt.legend()
plt.show()

'''
# Uncomment this part of the code when necessary
print("################################################################################################################")

# Accuracies for RealClass
print(accuracyHE1)
print(accuracyAHELC1)
print(accuracyGC1)
print(accuracyLT1)

print("################################################################################################################")

# Accuracies for ParentClass
print(accuracyHE2)
print(accuracyAHELC2)
print(accuracyGC2)
print(accuracyLT2)

print("################################################################################################################")

# Accuracies for GrandparentClass
print(accuracyHE3)
print(accuracyAHELC3)
print(accuracyGC3)
print(accuracyLT3)

print("################################################################################################################")

print(accuracyHE4)
print(accuracyAHELC4)
print(accuracyGC4)
print(accuracyLT4)
'''