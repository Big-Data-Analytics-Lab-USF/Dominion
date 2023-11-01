#Fuzzy Script
#Load packages
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#import dataframe of data to be compared
combinedData = pd.read_csv('/content/combined.csv', low_memory=False, encoding='latin-1')

#drop Na's and reset index
combinedData = combinedData.dropna(subset=['Title'])
combinedData.reset_index(drop=True, inplace=True)
#define function to process the dataframe with fuzzy string matching

def Fuzzy_de_duplicating(data, similairty_level):
  #define list to catch indexes to delete
  toDelete = []
  # Build loops to process data
  i = 0

  #check if iteration 'i' is already slated to be dropped from the dataframe
  for i, text1 in data.iterrows() :
    if i in toDelete :
      continue # skips over rows that have already been marked as too similar

  # next for loop to process rows that have not already been marked for deletion
    for index, text2 in data.iterrows():
      if index == i and index not in toDelete and index in data.index and index + 1 < data.shape[0] :
        str1 = str(text1['Title'])
        str2 = str(data.loc[index + 1, 'Title'])

      elif index != i and index not in toDelete and index in data.index : #check that index is not equal to current iteration and not in deletion list
        str1 = str(text1['Title'])
        str2 = str(text2['Title'])

      else :
        pass

    # Use fuzzy token ratio
      ratioScore = fuzz.token_sort_ratio(str1, str2)

      # if statment to test if two strings exceed a cetain similairty score
      if ratioScore >= similairty_level and index in data.index :
        # append indexes to list
        toDelete.append(index)


  data.drop(toDelete, inplace = True)

  return(data)


result = Fuzzy_de_duplicating(combinedData,90)

result.to_csv('/content/sample_data/completed_no_duplicates_90.csv')
