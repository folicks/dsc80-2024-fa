# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    # Create a list of tuples representing the DataFrame
    data = [
        ('Alice', 'Frank', 25),
        ('Bob', 'Grace', 30),
        ('Charlie', 'Henry', 35),
        ('David', 'Ivy', 40),
        ('Eve', 'Julia', 45)
    ]
    
    # Convert the list of tuples to a DataFrame
    tricky_1 = pd.DataFrame(data, columns=['Name', 'Name', 'Age'])
    
    # Save the DataFrame to a CSV file
    tricky_1.to_csv('tricky_1.csv', index=False)
    
    # Read the CSV file back into another DataFrame
    tricky_2 = pd.read_csv('tricky_1.csv')
    
    # Compare the DataFrames
    # print("Original DataFrame:")
    # print(tricky_1.columns)
    # print("\nRead-in DataFrame:")
    # print(tricky_2.columns)
    return 1


def trick_bool():
    return [10, 4, 13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------




def population_stats(df):
    # Calculate num_nonnull
    num_nonnull = df.count()
    
    # Calculate prop_nonnull
    prop_nonnull = num_nonnull / len(df)
    
    # Calculate num_distinct (excluding null values)
    num_distinct = df.nunique()
    
    # Calculate prop_distinct
    prop_distinct = num_distinct / num_nonnull
    
    # Create the result DataFrame
    result = pd.DataFrame({
        'num_nonnull': num_nonnull,
        'prop_nonnull': prop_nonnull,
        'num_distinct': num_distinct,
        'prop_distinct': prop_distinct
    })
    
    return result
# # Create a sample DataFrame
# df = pd.DataFrame({
#     'ages': [2, 2, 2, np.nan, 5, 7, 5, 10, 11, np.nan]
# })

# # Call the function
# result = population_stats(df)

# # Print the result
# (result)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(df, N):
    # Initialize an empty DataFrame with N rows and index 0 to N-1
    result = pd.DataFrame(index=range(N))
    
    for column in df.columns:
        # Get value counts for the column
        value_counts = df[column].value_counts()
        
        # Get the N most common values and their counts
        top_N = value_counts.iloc[:N]
        
        # Create a DataFrame for this column
        column_df = pd.DataFrame({
            f'{column}_values': top_N.index,
            f'{column}_counts': top_N.values
        })
        
        # If there are fewer than N distinct values, fill with NaN
        if len(column_df) < N:
            column_df = column_df.reindex(range(N), fill_value=np.nan)
        
        # Add the column data to the result DataFrame
        result[f'{column}_values'] = column_df[f'{column}_values']
        result[f'{column}_counts'] = column_df[f'{column}_counts']
    
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    # 1. Superhero with the greatest number of superpowers
    most_powers_name = powers.loc[powers.iloc[:, 1:].sum(axis=1).idxmax(), 'hero_names']

    # 2. Most common superpower among flying superheroes (excluding 'Flight')
    flying_heroes = powers[powers['Flight'] == True]
    common_flying_power = flying_heroes.drop(['hero_names', 'Flight'], axis=1).sum().idxmax()

    # 3. Most common superpower among heroes with only one superpower
    single_power_heroes = powers[powers.iloc[:, 1:].sum(axis=1) == 1]
    common_single_power = single_power_heroes.iloc[:, 1:].sum().idxmax()

    return [most_powers_name, common_flying_power, common_single_power]
# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(df):
    #TODO
        # replace instances of "-" in Skin color with np.nan
        # GOAL do so in one line
    
    # return df.replace(['-', 'null', ''], np.nan)
    return df.replace(['-', 'null', ''], np.nan).mask((df['Height'] < 0) | (df['Weight'] < 0))


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return [
        'Onslaught',
        'Dark Horse Comics',
        'bad',
        'Marvel Comics',
        'Dark Horse Comics',
        'Groot'
    ]



# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------



def clean_universities(df):
    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    
    # Replace '\n' with ', ' in the 'institution' column
    cleaned_df['institution'] = cleaned_df['institution'].str.replace('\n', ', ')
    
    # Change the data type of 'broad_impact' to int
    cleaned_df['broad_impact'] = cleaned_df['broad_impact'].astype(int)
    
    # Split 'national_rank' into 'nation' and 'national_rank_cleaned'
    cleaned_df[['nation', 'national_rank_cleaned']] = cleaned_df['national_rank'].str.split(', ', expand=True)
    
    # Replace country names
    country_replacements = {
        'Czechia': 'Czech Republic',
        'UK': 'United Kingdom',
        'USA': 'United States'
    }
    cleaned_df['nation'] = cleaned_df['nation'].replace(country_replacements)
    
    # Convert 'national_rank_cleaned' to int
    cleaned_df['national_rank_cleaned'] = cleaned_df['national_rank_cleaned'].astype(int)
    
    # Drop the original 'national_rank' column
    cleaned_df = cleaned_df.drop('national_rank', axis=1)
    
    # Create 'is_r1_public' column
    cleaned_df['is_r1_public'] = (
        (cleaned_df['control'] == 'Public') &
        cleaned_df['city'].notna() &
        cleaned_df['state'].notna()
    )
    
    # Ensure 'is_r1_public' is False for NaN values
    cleaned_df['is_r1_public'] = cleaned_df['is_r1_public'].fillna(False)
    
    return cleaned_df


def university_info(df):
    result = []

    # 1. State with lowest mean score (among states with 3+ institutions)
    state_counts = df['state'].value_counts()
    states_with_3_plus = state_counts[state_counts >= 3].index
    state_mean_scores = df[df['state'].isin(states_with_3_plus)].groupby('state')['score'].mean()
    lowest_score_state = state_mean_scores.idxmin()
    result.append(lowest_score_state)

    # 2. Proportion of top 100 institutions also in top 100 for quality of faculty
    top_100 = df[df['world_rank'] <= 100]
    prop_top_100_faculty = (top_100['quality_of_faculty'] <= 100).mean()
    result.append(prop_top_100_faculty)

    # 3. Number of states where at least 50% of institutions are private
    state_private_ratio = df.groupby('state')['is_r1_public'].apply(lambda x: (x == False).mean())
    states_majority_private = (state_private_ratio >= 0.5).sum()
    result.append(states_majority_private)

    # 4. Lowest world-ranked institution that is highest-ranked in its nation
    top_national = df[df['national_rank_cleaned'] == 1]
    worst_top_national = top_national.loc[top_national['world_rank'].idxmax(), 'institution']
    result.append(worst_top_national)

    return result    

