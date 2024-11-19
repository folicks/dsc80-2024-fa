# lab.py


import pandas as pd
import numpy as np
import io
from pathlib import Path
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------



def prime_time_logins(login_df):
    '''
    Takes in a DataFrame with 'Login Id' and 'Time' columns,
    and returns a DataFrame with the count of prime-time logins for each user.
    Prime-time is defined as between 4PM (inclusive) and 8PM (exclusive).
    '''
    # Make a copy of the DataFrame to avoid modifying the original
    login_df = login_df.copy()

    # Convert 'Time' column to datetime
    login_df['Time'] = pd.to_datetime(login_df['Time'], format='%Y-%m-%d %H:%M:%S')
    
    # Extract the hour component from the datetime values
    hours = login_df['Time'].dt.hour
    
    # Create a mask for prime-time logins (4PM to 8PM)
    prime_time_mask = (hours >= 16) & (hours < 20)
    
    # Count prime-time logins for each user
    prime_time_count = login_df[prime_time_mask].groupby('Login Id').size().to_frame('Time')
    
    # Ensure all users are included, even those with zero prime-time logins
    all_users = login_df['Login Id'].unique()
    prime_time_count = prime_time_count.reindex(all_users, fill_value=0)
    
    return prime_time_count


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def count_frequency(login_df):
    # Make a copy of the DataFrame to avoid modifying the original
    login_df = login_df.copy()
    
    # Convert 'Time' column to datetime
    login_df['Time'] = pd.to_datetime(login_df['Time'])
    
    # Set the end date to January 31st, 2024, at 11:59 PM
    end_date = pd.Timestamp('2024-01-31 23:59:00')
    
    # Define a custom aggregation function to calculate login frequency
    def login_frequency(group):
        first_login = group.min()  # First login date per user
        total_logins = group.size  # Use 'size' as a property, not a method
        days_as_member = (end_date - first_login).days  # Days as member (inclusive)
        if days_as_member == 0:
            return total_logins  # Handle users with logins only on one day
        return total_logins / (days_as_member)  # +1 to include the last day
    
    # Group by 'Login Id' and apply the custom aggregation
    result = login_df.groupby('Login Id')['Time'].agg(login_frequency)
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cookies_null_hypothesis():
    """
    Returns a list of reasonable choices for the null hypothesis.
    """
    return [1, 2]

def cookies_p_value(N_simulations):
    """
    Estimates the p-value of the investigation by simulating the null hypothesis N times.
    
    Parameters:
    N_simulations (int): The number of simulations to run.
    
    Returns:
    float: The estimated p-value.
    """
    # Define the parameters of the null hypothesis
    n_cookies = 250
    p_burnt = 0.04
    
    # Simulate the null hypothesis N times
    simulations = np.random.binomial(n_cookies, p_burnt, N_simulations)
    
    # Calculate the observed statistic (number of burnt cookies)
    observed_statistic = 15
    
    # Calculate the p-value (one-tailed test)
    p_value = np.mean(simulations >= observed_statistic)
    
    return float(p_value)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def car_null_hypothesis():
    # The claim is that tires stop under 106 feet exactly 95% of the time
    return [1]  # Option 1

def car_alt_hypothesis():
    # We suspect the success rate is less than 95%
    return [2]  # Option 2

def car_test_statistic():  # Fixed function name to match test case
    # Both count and proportion are valid test statistics
    return [1, 4]  # Options 1 and 4

def car_p_value():
    # The probability of any exact value becomes tiny with more trials
    return 4  # Option 4


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def superheroes_test_statistic():
    # The valid test statistics are:
    # 1. Difference in proportions (Option 1)
    # 4. Absolute difference in proportions (Option 4)
    return [1, 4]

def bhbe_col(df):
    """Return a Boolean Series for blond-haired, blue-eyed characters."""
    return (df['Hair color'].str.contains('blond', case=False, na=False)) & \
           (df['Eye color'].str.contains('blue', case=False, na=False))

def superheroes_observed_statistic(df):
    """Calculate the observed proportion of 'good' characters among blond-haired, blue-eyed characters."""
    bhbe = bhbe_col(df)
    good_bhbe = (df.loc[bhbe, 'Alignment'] == 'good').mean()
    return good_bhbe

def simulate_bhbe_null(df, N):
    """Simulate N test statistics under the null hypothesis."""
    bhbe = bhbe_col(df)
    good_proportion = (df['Alignment'] == 'good').mean()
    # Simulate the number of 'good' characters in the bhbe group under the null
    simulated_stats = np.random.binomial(n=bhbe.sum(), p=good_proportion, size=N) / bhbe.sum()
    return simulated_stats

def superheroes_p_value(df):
    """Compute the p-value and decide whether to reject the null hypothesis."""
    observed_stat = superheroes_observed_statistic(df)
    simulated_stats = simulate_bhbe_null(df, 100000)
    
    # Calculate the p-value based on the proportion of simulations >= observed_stat
    p_value = (simulated_stats >= observed_stat).mean()
    
    # Determine the hypothesis test result based on the 1% significance level
    decision = 'Reject' if p_value < 0.01 else 'Fail to reject'
    
    # Ensure the p-value is returned as a standard Python float, not np.float64
    return [float(p_value), decision]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def diff_of_means(data, col='orange'):
    yorkville_mean = data[data['Factory'] == 'Yorkville'][col].mean()
    waco_mean = data[data['Factory'] == 'Waco'][col].mean()
    return abs(yorkville_mean - waco_mean)

def simulate_null(data, col='orange'):
    shuffled_data = data.copy()
    shuffled_data['Factory'] = np.random.permutation(shuffled_data['Factory'])
    return diff_of_means(shuffled_data, col)

def color_p_value(data, col='orange', num_trials=1000):
    observed_diff = diff_of_means(data, col)
    simulated_diffs = [simulate_null(data, col) for _ in range(num_trials)]
    p_value = np.mean(np.array(simulated_diffs) >= observed_diff)
    return p_value


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def ordered_colors():
    """Return a hard-coded list of ('color', p_value) pairs, sorted by p-value."""
    return [
        ('orange', (0.056)),
        ('yellow', (0.0)),
        ('red', (0.24)),
        ('green', (0.476)),
        ('purple', (0.976))
    ]


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


    

def calculate_tvd(p, q):
    """Calculate the Total Variation Distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))

def same_color_distribution():
    # Load the data and ensure correct parsing
    data = pd.read_csv('data/skittles.tsv', sep='\t')
    
    # Check column names and clean up any whitespace
    data.columns = data.columns.str.strip()

    # Aggregate the counts of each color by factory
    yorkville_totals = data[data['Factory'] == 'Yorkville'].iloc[:, :-1].sum()
    waco_totals = data[data['Factory'] == 'Waco'].iloc[:, :-1].sum()

    # Compute proportions of each color for both factories
    yorkville_proportions = yorkville_totals / yorkville_totals.sum()
    waco_proportions = waco_totals / waco_totals.sum()

    # Calculate the observed TVD between the two factories
    observed_tvd = calculate_tvd(yorkville_proportions, waco_proportions)

    # Perform permutation test with 1000 trials
    n_trials = 1000
    tvd_values = np.zeros(n_trials)

    for i in range(n_trials):
        # Shuffle the factory labels
        shuffled_labels = np.random.permutation(data['Factory'])

        # Assign new factory labels and aggregate by the shuffled labels
        yorkville_shuffled_totals = data[shuffled_labels == 'Yorkville'].iloc[:, :-1].sum()
        waco_shuffled_totals = data[shuffled_labels == 'Waco'].iloc[:, :-1].sum()

        # Compute new proportions for shuffled data
        yorkville_shuffled_prop = yorkville_shuffled_totals / yorkville_shuffled_totals.sum()
        waco_shuffled_prop = waco_shuffled_totals / waco_shuffled_totals.sum()

        # Compute TVD for the shuffled data
        tvd_values[i] = calculate_tvd(yorkville_shuffled_prop, waco_shuffled_prop)

    # Calculate the p-value
    p_value = (tvd_values >= observed_tvd).mean()

    # Determine whether to reject the null hypothesis
    decision = 'Reject' if p_value < 0.01 else 'Fail to reject'

    # Return a hard-coded result (rounded p-value and decision)
    return (round(p_value, 3), decision)

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def perm_vs_hyp():
    return ['P', 'P', 'P', 'H', 'P']

