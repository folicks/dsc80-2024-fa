# lab.py


from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def after_purchase():
    return ['NMAR','MD', 'MAR', 'MAR', 'MAR']


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multiple_choice():
    return ["MAR", "MAR", "MAR", "NMAR", "MCAR"]


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def calculate_ages(df):
    """Calculate ages from date_of_birth column, relative to 2024."""
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%d-%b-%Y', errors='coerce')
    reference_date = pd.Timestamp('2024-01-01')
    ages = (reference_date - df['date_of_birth']).dt.days / 365.25
    return ages

def get_age_groups(df):
    """Split ages into groups based on credit_card_number missingness."""
    ages = calculate_ages(df)
    is_missing = df['credit_card_number'].isna()
    
    missing_ages = ages[is_missing].dropna()
    present_ages = ages[~is_missing].dropna()
    
    return missing_ages, present_ages

def mean_diff_statistic(group1, group2):
    """Calculate absolute difference of means between two groups."""
    return abs(np.mean(group1) - np.mean(group2))

def run_permutation_test(group1, group2, statistic_func, n_permutations=10000):
    """Run a permutation test using the provided statistic function."""
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    observed_stat = statistic_func(group1, group2)
    
    permuted_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_stat = statistic_func(perm_group1, perm_group2)
        permuted_stats.append(perm_stat)
    
    # Calculate p-value and convert to float
    p_value = float(np.mean(np.array(permuted_stats) >= observed_stat))
    return p_value

def first_round():
    """Perform first round analysis using mean difference statistic."""
    df = pd.read_csv('data/payment.csv')
    
    missing_ages, present_ages = get_age_groups(df)
    
    p_value = run_permutation_test(missing_ages, present_ages, mean_diff_statistic)
    
    result = 'R' if p_value < 0.05 else 'NR'
    
    return [float(p_value), result]

def second_round():
    """Perform second round analysis using KS statistic."""
    df = pd.read_csv('data/payment.csv')
    
    missing_ages, present_ages = get_age_groups(df)
    
    # Run KS test and convert p-value to float
    ks_stat, p_value = stats.ks_2samp(missing_ages, present_ages)
    p_value = float(p_value)
    
    result = 'R' if p_value < 0.05 else 'NR'
    conclusion = 'D' if result == 'R' else 'ND'
    
    return [p_value, result, conclusion]

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def verify_child(heights):
    """
    Verify MAR dependency between father's heights and missingness in child_X columns
    using Kolmogorov-Smirnov tests.
    
    Args:
        heights (pd.DataFrame): DataFrame containing father and child_X height columns
        
    Returns:
        pd.Series: Series of p-values from KS tests, indexed by child_X column names
    """
    # Get all child_X columns
    child_cols = [col for col in heights.columns if col.startswith('child_') and col != 'child']
    
    # Dictionary to store p-values
    p_values = {}
    
    for col in child_cols:
        # Split father heights based on missingness in child_X column
        father_heights_missing = heights.loc[heights[col].isna(), 'father']
        father_heights_present = heights.loc[~heights[col].isna(), 'father']
        
        # Perform KS test between the two distributions
        # Using stats.ks_2samp which returns both statistic and p-value
        _, p_value = stats.ks_2samp(father_heights_missing, father_heights_present)
        
        # Store p-value
        p_values[col] = p_value
    
    # Convert to Series and sort by column names
    return pd.Series(p_values)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def cond_single_imputation(heights):
    """
    Perform single-valued mean imputation of child heights conditional on father height quartiles.
    
    Args:
        heights (pd.DataFrame): DataFrame with 'father' and 'child' columns
        
    Returns:
        pd.Series: Series with imputed child heights
    """
    # Create father height quartiles
    heights['father_quartile'] = pd.qcut(heights['father'], q=4)
    
    # Calculate mean child height for each father height quartile
    # transform() will apply the mean to each group and broadcast back to original size
    imputed_values = heights.groupby('father_quartile')['child'].transform(lambda x: x.fillna(x.mean()))
    
    return imputed_values


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    Generate N imputed values based on the distribution of observed values.
    
    Args:
        child (pd.Series): Series of child heights with missing values
        N (int): Number of values to generate
        
    Returns:
        np.array: Array of N imputed values
    """
    # Get observed (non-missing) values
    observed = child.dropna()
    
    # Create histogram with 10 bins
    counts, bin_edges = np.histogram(observed, bins=10)
    
    # Calculate probabilities for each bin
    probabilities = counts / len(observed)
    
    # Generate imputed values
    imputed_values = np.zeros(N)
    
    # For each value we need to generate
    for i in range(N):
        # Randomly select a bin based on probabilities
        selected_bin = np.random.choice(len(counts), p=probabilities)
        
        # Generate a uniform random value within the selected bin
        bin_min = bin_edges[selected_bin]
        bin_max = bin_edges[selected_bin + 1]
        imputed_values[i] = np.random.uniform(bin_min, bin_max)
    
    return imputed_values

def impute_height_quant(child):
    """
    Impute missing values in child heights using quantitative distribution method.
    
    Args:
        child (pd.Series): Series of child heights with missing values
        
    Returns:
        pd.Series: Series with imputed values
    """
    # Count number of missing values
    n_missing = child.isna().sum()
    
    # Generate imputed values
    imputed_values = quantitative_distribution(child, n_missing)
    
    # Create copy of input series
    result = child.copy()
    
    # Replace missing values with generated values
    result[result.isna()] = imputed_values
    
    return result


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------

def answers():
    return [1,2,2,1],["https://www.reddit.com/robots.txt","https://dsc10.com/"]
