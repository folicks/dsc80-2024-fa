# project.py


import pandas as pd
import numpy as np
from pathlib import Path

##my change
import json

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    """
    Clean the loans DataFrame according to specified requirements.
    
    Parameters:
    loans (pd.DataFrame): Input DataFrame containing loan information
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with transformed columns
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned = loans.copy()
    
    # Convert issue_d to datetime
    cleaned['issue_d'] = pd.to_datetime(cleaned['issue_d'], format='%b-%Y')
    
    # Clean term column: extract the number of months as integers
    cleaned['term'] = cleaned['term'].str.extract('(\\d+)').astype(int)
    
    # Clean emp_title column
    # First, convert to lowercase and strip whitespace
    cleaned['emp_title'] = cleaned['emp_title'].str.lower().str.strip()
    
    # Replace exact matches of 'rn' with 'registered nurse'
    cleaned.loc[cleaned['emp_title'] == 'rn', 'emp_title'] = 'registered nurse'
    
    # Calculate term_end dates
    cleaned['term_end'] = cleaned['issue_d'].apply(
        lambda x: x + pd.DateOffset(months=int(cleaned.loc[cleaned['issue_d'] == x, 'term'].iloc[0]))
    )
    
    return cleaned

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------




def correlations(df, pairs):
    """
    Calculate Pearson correlations between specified pairs of columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    pairs (list of tuples): List of (col1, col2) tuples specifying column pairs to correlate
    
    Returns:
    pd.Series: Correlations with index of form 'r_col1_col2'
    """
    # Initialize empty dictionary to store results
    corr_dict = {}
    
    # Calculate correlation for each pair
    for col1, col2 in pairs:
        # Create index name in required format
        index_name = f'r_{col1}_{col2}'
        
        # Calculate correlation using pandas corr() method
        correlation = df[col1].corr(df[col2])
        
        # Store in dictionary
        corr_dict[index_name] = correlation
    
    # Convert dictionary to Series
    return pd.Series(corr_dict)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(df):
    # Bin the credit scores into the specified ranges
    bins = [580, 670, 740, 800, 850]
    labels = ['[580, 670)', '[670, 740)', '[740, 800)', '[800, 850)']
    df['fico_bin'] = pd.cut(df['fico_range_low'], bins=bins, labels=labels, right=False)

    # Create the boxplot
    fig = px.box(
        df, 
        x='fico_bin',
        y='int_rate',
        color='term',
        color_discrete_map={'36':'purple', '60':'gold'},
        title='Interest Rate vs. Credit Score',
        labels={
            'fico_bin': 'Credit Score Range',
            'int_rate': 'Interest Rate (%)',
            'term': 'Loan Length (Months)'
        }
    )

    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(df, n_repetitions=5000):
    # Split the data into those with and without personal statements
    has_ps = df['desc'].notna()
    
    # Calculate the observed difference in mean interest rates
    obs_diff = df.loc[has_ps, 'int_rate'].mean() - df.loc[~has_ps, 'int_rate'].mean()
    
    # Initialize a counter for extreme differences
    extreme_count = 0
    
    # Perform the permutation test
    for _ in range(n_repetitions):
        # Shuffle the desc column
        shuffled_desc = np.random.permutation(df['desc'].notna().values)
        
        # Recalculate the mean difference with the shuffled data
        shuffled_diff = df.loc[shuffled_desc, 'int_rate'].mean() - df.loc[~shuffled_desc, 'int_rate'].mean()
        
        # Check if the shuffled difference is as extreme as or more extreme than the observed difference
        if shuffled_diff >= obs_diff:
            extreme_count += 1
    
    # Calculate the p-value
    p_value = extreme_count / n_repetitions
    
    return p_value
    
def missingness_mechanism():
    return 1
    
def argument_for_nmar():
    return '''
    The people asking for loans who feel the need to make a statement may \
    just so happen to be risky borrowers to begin with or in other they aren't becoming risky because of the statement\
    they just already were
    '''


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    """
    Calculate total tax owed based on income and tax brackets.
    
    Args:
        income (float): Gross income amount
        brackets (list): List of tuples [(rate, lower_limit), ...] representing tax brackets
        
    Returns:
        float: Total tax owed, rounded to 2 decimal places
    """
    total_tax = 0
    
    for i in range(len(brackets)):
        current_rate, current_lower = brackets[i]
        
        # Get the upper limit of current bracket (lower limit of next bracket, or income if it's the last bracket)
        current_upper = brackets[i + 1][1] if i < len(brackets) - 1 else float('inf')
        
        if income <= current_lower:
            # Income is below this bracket
            break
        elif income > current_lower:
            # Calculate taxable amount for this bracket
            taxable_amount = min(income - current_lower, current_upper - current_lower)
            total_tax += taxable_amount * current_rate
    
    return total_tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(df):
    def clean_state_column(df):
        # Keep only valid state names, replace others with NaN
        df['State'] = df['State'].where(df['State'].str.contains(r'^[A-Z][a-z]{1,}\.?$', na=False), np.nan)
        # Forward fill state names
        df['State'] = df['State'].ffill()
        return df

    def clean_rate_column(df):
        # Convert rates to float, handle 'none' case
        df['Rate'] = df['Rate'].replace('none', '0%')
        df['Rate'] = df['Rate'].str.rstrip('%').astype(float) / 100
        df['Rate'] = df['Rate'].round(4)  # Round to 4 decimal places
        return df

    def clean_lower_limit_column(df):
        # Convert lower limits to integer, handle NaN case
        df['Lower Limit'] = df['Lower Limit'].fillna(0)
        df['Lower Limit'] = df['Lower Limit'].replace(r'[\$,]', '', regex=True).astype(int)
        return df

    return (df
            .dropna(how='all')  # Drop rows that are all NaN
            .pipe(clean_state_column)
            .pipe(clean_rate_column)
            .pipe(clean_lower_limit_column)
            .reset_index(drop=True))  # Reset index after cleaning


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(df):
    def create_bracket_list(group):
        return list(zip(group['Rate'], group['Lower Limit']))
    
    return df.groupby('State').apply(create_bracket_list).reset_index(name='bracket_list')

def combine_loans_and_state_taxes(loans, state_taxes):
    # Load the state mapping
    with open('data/state_mapping.json', 'r') as f:
        state_mapping = json.load(f)
    
    # Reverse the mapping for our use
    reverse_mapping = {v: k for k, v in state_mapping.items()}
    
    # Create the state brackets
    brackets = state_brackets(state_taxes)
    
    # Map the state names in brackets to two-letter abbreviations
    brackets['State'] = brackets['State'].map(state_mapping)
    
    # Rename the 'addr_state' column in loans to 'State'
    loans = loans.rename(columns={'addr_state': 'State'})
    
    # Merge the DataFrames
    result = loans.merge(brackets, on='State', how='left')
    
    # For states not in the tax data, set an empty bracket list
    result['bracket_list'] = result['bracket_list'].fillna(result['State'].map(
        lambda x: [(0.0, 0)] if x not in brackets['State'].values else None
    ))
    
    return result


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(df):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]

    def calculate_federal_tax(income):
        return tax_owed(income, FEDERAL_BRACKETS)

    def calculate_state_tax(income, brackets):
        return tax_owed(income, brackets)

    # Create a copy of the input DataFrame
    result = df.copy()

    # Calculate federal tax owed
    result['federal_tax_owed'] = result['annual_inc'].apply(calculate_federal_tax)

    # Calculate state tax owed
    result['state_tax_owed'] = result.apply(lambda row: calculate_state_tax(row['annual_inc'], row['bracket_list']), axis=1)

    # Calculate disposable income
    result['disposable_income'] = result['annual_inc'] - result['federal_tax_owed'] - result['state_tax_owed']

    return result


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    """
    Aggregate loan data by keywords in job titles and combine results.
    
    Args:
        loans (DataFrame): DataFrame containing loan data
        keywords (list): List of two strings to search for in job titles
        quantitative_column (str): Name of quantitative column to calculate means for
        categorical_column (str): Name of categorical column to group by
    
    Returns:
        DataFrame: Aggregated means by category and overall for each keyword
    """
    # Initialize empty dictionary to store results
    results = {}
    
    # Process each keyword
    for keyword in keywords:
        # Create column name using f-string
        col_name = f'{keyword}_mean_{quantitative_column}'
        
        # Filter for rows where keyword appears in emp_title
        keyword_mask = loans['emp_title'].str.contains(keyword, na=False)
        filtered_df = loans[keyword_mask]
        
        # Calculate means by category
        category_means = filtered_df.groupby(categorical_column)[quantitative_column].mean()
        
        # Calculate overall mean
        overall_mean = filtered_df[quantitative_column].mean()
        
        # Combine category means and overall mean
        all_means = pd.concat([
            category_means,
            pd.Series({'Overall': overall_mean})
        ])
        
        # Store in results dictionary
        results[col_name] = all_means
    
    # Combine results into final DataFrame
    final_df = pd.DataFrame(results)
    
    # Ensure 'Overall' is the last row
    non_overall = final_df.index != 'Overall'
    final_df = pd.concat([
        final_df[non_overall],
        final_df[~non_overall]
    ])
    
    return final_df

# Example usage:
# aggregate_and_combine(loans, ['engineer', 'nurse'], 'loan_amnt', 'home_ownership')


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    """
    Check if Simpson's Paradox exists in the aggregated data.
    
    Args:
        loans (DataFrame): Loan data
        keywords (list): Two job title keywords to compare
        quantitative_column (str): Column to calculate means for
        categorical_column (str): Column to group by
    
    Returns:
        bool: True if Simpson's Paradox exists, False otherwise
    """
    df = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    col1, col2 = df.columns
    # Check if all category-wise comparisons show opposite trend from overall
    return ((df[col1] > df[col2]).iloc[:-1].all() and df[col1].iloc[-1] < df[col2].iloc[-1]) or \
           ((df[col1] < df[col2]).iloc[:-1].all() and df[col1].iloc[-1] > df[col2].iloc[-1])

    
def paradox_example(loans):
    """
    Find an example of Simpson's Paradox in the loan data.
    
    Args:
        loans (DataFrame): Loan data
    
    Returns:
        dict: Dictionary with parameters that demonstrate Simpson's Paradox
    """
    # Let's compare teachers with managers and look at annual income grouped by grade
    # This combination should show a different paradox than the engineer/nurse example
    return {
        'loans': loans,
        'keywords': ['manager', 'engineer'],
        'quantitative_column': 'annual_inc',
        'categorical_column': 'sub_grade'
    }
