# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("The specified directory does not exist")
    
    dataframes = []
    for file in directory.iterdir():
        if file.name.startswith('survey') and file.suffix == '.csv':
            df = pd.read_csv(file)
            # Ensure we have the correct columns
            required_columns = ['first name', 'last name', 'current company', 'job title', 'email', 'university']
            if set(required_columns).issubset(df.columns):
                df = df[required_columns]
            else:
                # If columns don't match, try to infer them based on position
                df.columns = required_columns[:len(df.columns)]
            dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No survey files found in the specified directory")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Ensure we have all required columns, fill with NaN if missing
    for col in required_columns:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA
    
    # Reorder columns to match the required order
    combined_df = combined_df[required_columns]
    
    return combined_df.reset_index(drop=True)

def com_stats(df):
    ohio_programmer = df[(df['university'].str.contains('Ohio')) & (df['job title'].str.contains('Programmer'))].shape[0] / df.shape[0]
    
    engineer_titles = df['job title'].str.endswith('Engineer').sum()
    
    longest_title = df['job title'].str.len().idxmax()
    longest_title = df.loc[longest_title, 'job title']
    
    manager_count = df['job title'].str.contains('manager', case=False).sum()
    
    return [ohio_programmer, engineer_titles, longest_title, manager_count]



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("The specified directory does not exist")
    
    dataframes = []
    for file in directory.iterdir():
        if file.name.startswith('favorite') and file.suffix == '.csv':
            df = pd.read_csv(file, index_col='id')
            dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No survey files found in the specified directory")
    
    combined_df = pd.concat(dataframes, axis=1)
    
    # Ensure 'name' column is present (from favorite1.csv)
    if 'name' not in combined_df.columns:
        raise ValueError("Missing 'name' column in survey data")
    
    return combined_df


def check_credit(df):
    # Calculate the number of questions each student answered
    questions_answered = df.notna().sum(axis=1) - 1  # Subtract 1 to exclude 'name' column
    total_questions = df.shape[1] - 1  # Subtract 1 to exclude 'name' column
    
    # Calculate individual extra credit
    individual_ec = (questions_answered >= 0.5 * total_questions).astype(int) * 5
    
    # Calculate class-wide extra credit
    response_rate = df.notna().mean()
    high_response_questions = (response_rate >= 0.9).sum() - 1  # Subtract 1 to exclude 'name' column
    class_ec = min(high_response_questions, 2)
    
    # Combine individual and class extra credit
    total_ec = individual_ec + class_ec
    
    # Create the result DataFrame
    result = pd.DataFrame({
        'name': df['name'],
        'ec': total_ec
    })
    
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets: pd.DataFrame, procedure_history: pd.DataFrame) -> str:
    """
    Find the most common ProcedureType across all pets.
    
    Args:
        pets: DataFrame containing pet information
        procedure_history: DataFrame containing procedure records
    
    Returns:
        str: The name of the most common ProcedureType
    """
    # Count occurrences of each ProcedureType
    procedure_counts = procedure_history['ProcedureType'].value_counts()
    
    # Return the most common procedure
    return procedure_counts.index[0]

def pet_name_by_owner(owners: pd.DataFrame, pets: pd.DataFrame) -> pd.Series:
    """
    Create a Series mapping owner first names to their pets' names.
    
    Args:
        owners: DataFrame containing owner information
        pets: DataFrame containing pet information
    
    Returns:
        pd.Series: Series with owner first names as index and pet names (string or list) as values
    """
    # Merge owners and pets DataFrames using left merge to keep all owners
    merged_df = pd.merge(owners[['OwnerID', 'Name']], pets[['OwnerID', 'Name']], 
                        on='OwnerID', how='left', suffixes=('_owner', '_pet'))
    
    # Group by owner name and aggregate pet names
    def aggregate_pets(x):
        # Remove NaN values
        valid_pets = x.dropna()
        if len(valid_pets) == 0:
            return None
        elif len(valid_pets) == 1:
            return valid_pets.iloc[0]
        else:
            return list(valid_pets)
    
    result = merged_df.groupby('Name_owner')['Name_pet'].agg(aggregate_pets)
    
    return result

def total_cost_per_city(owners: pd.DataFrame, pets: pd.DataFrame, 
                       procedure_history: pd.DataFrame, procedure_detail: pd.DataFrame) -> pd.Series:
    """
    Calculate the total amount spent on procedures per city.
    
    Args:
        owners: DataFrame containing owner information
        pets: DataFrame containing pet information
        procedure_history: DataFrame containing procedure records
        procedure_detail: DataFrame containing procedure information and prices
    
    Returns:
        pd.Series: Series with cities as index and total costs as values
    """
    # Merge procedure_history with procedure_detail to get prices
    procedures_with_prices = pd.merge(
        procedure_history,
        procedure_detail[['ProcedureType', 'ProcedureSubCode', 'Price']],
        on=['ProcedureType', 'ProcedureSubCode']
    )
    
    # Merge with pets to get OwnerID
    procedures_with_owners = pd.merge(
        procedures_with_prices,
        pets[['PetID', 'OwnerID']],
        on='PetID',
        how='left'
    )
    
    # Merge with owners to get City
    procedures_by_city = pd.merge(
        procedures_with_owners,
        owners[['OwnerID', 'City']],
        on='OwnerID',
        how='right'
    )
    
    # Calculate total cost per city
    city_costs = procedures_by_city.groupby('City')['Price'].sum().fillna(0)
    
    return city_costs


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


import pandas as pd

def average_seller(sales):
    """
    Calculate the average sales for each seller.
    
    Args:
        sales (pd.DataFrame): DataFrame containing sales data with 'Name' and 'Total' columns
    
    Returns:
        pd.DataFrame: DataFrame indexed by 'Name' with 'Average Sales' column
    """
    return sales.groupby('Name')['Total'].mean().to_frame('Average Sales')

def product_name(sales):
    """
    Calculate total sales for each product by seller.
    
    Args:
        sales (pd.DataFrame): DataFrame containing sales data with 'Name', 'Product', and 'Total' columns
    
    Returns:
        pd.DataFrame: Pivot table showing total sales per product for each seller
    """
    return pd.pivot_table(
        sales,
        values='Total',
        index='Name',
        columns='Product',
        aggfunc='sum'
    )

def count_product(sales):
    """
    Count number of items sold by product and seller per date.
    
    Args:
        sales (pd.DataFrame): DataFrame containing sales data with 'Product', 'Name', 
                            'Date' columns
    
    Returns:
        pd.DataFrame: Pivot table showing count of items sold, with Product and Name as index
                     and Date as columns
    """
    pivot_df = pd.pivot_table(
        sales,
        values='Total',
        index=['Product', 'Name'],
        columns=['Date'],
        aggfunc='count',
        fill_value=0
    )
    return pivot_df

def total_by_month(sales):
    """
    Calculate total sales by seller and product per month.
    
    Args:
        sales (pd.DataFrame): DataFrame containing sales data with 'Name', 'Product', 
                            'Date', and 'Total' columns
    
    Returns:
        pd.DataFrame: Pivot table showing total sales with Name and Product as index
                     and Month as columns
    """
    # Extract month from date
    sales['Month'] = pd.to_datetime(sales['Date']).dt.strftime('%B')
    
    pivot_df = pd.pivot_table(
        sales,
        values='Total',
        index=['Name', 'Product'],
        columns=['Month'],
        aggfunc='sum',
        fill_value=0
    )
    return pivot_df