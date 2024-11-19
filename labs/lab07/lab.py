# lab.py


import pandas as pd
import numpy as np
import os
import re

import doctest

# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    # pattern = ".{2}\[.]{1}\\]"
    # pattern = r'^..[[]..[]$'
    # pattern = ".{2}\[.\]\."
    # pattern = ".{2}\[.\].*"
    # pattern = "^..\[.\].*$"
    pattern = ".{2}\[.\]"


    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    # pattern = '\(858\) \d{3} \d{3}-\d{4}'
    # pattern = r'^$858$ \d{3} \d{3}-\d{4}$'
    pattern = "^\(858\) \d{3}-\d{4}$"


    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    # pattern = r'^[\w ?]{6,10}\?$'
    pattern = "^[\w\s?]{6,10}$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r'^\$[^abc\$]*\$[aA]+[bB]+[cC]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r'^\w+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = "^[a-z]+_[a-z]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = "^_.*_$"

    
    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = "^[^Oi1]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = "^(NY-\d{2}-[A-Z]{3}|CA-\d{2}-(SAN|LAX))-\d{4}$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    s = string.lower()
    # Remove non-alphanumeric and 'a'
    s = re.sub('[^a-z0-9]|a', '', s)
    # Get three-character substrings
    return re.findall('...', s)

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_personal(text):
    """
    Extracts personal information from messy server log data.
    
    Args:
        text (str): Raw server log content
    
    Returns:
        tuple: (emails, ssns, bitcoin_addresses, street_addresses)
    """
    # Initialize lists to store extracted information
    emails = []
    ssns = []
    bitcoin_addresses = []
    street_addresses = []
    
    # Regular expression patterns
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    ssn_pattern = r'ssn:(\d{3}-\d{2}-\d{4})'
    bitcoin_pattern = r'bitcoin:([1-9A-HJ-NP-Za-km-z]{26,35})'
    # Street address pattern looks for number followed by words ending with common street types
    street_pattern = r'\d+[A-Za-z\s]+(Street|Road|Avenue|Drive|Lane|Court|Park|Place|Way|Circle|Boulevard|Blvd|Ave|Dr|Ln|Ct|Rd)\b'
    
    # Extract emails
    emails = re.findall(email_pattern, text)
    
    # Extract SSNs (excluding 'null' values)
    ssn_matches = re.findall(ssn_pattern, text)
    ssns = [ssn for ssn in ssn_matches if ssn != 'null']
    
    # Extract Bitcoin addresses
    bitcoin_matches = re.findall(bitcoin_pattern, text)
    bitcoin_addresses = [addr for addr in bitcoin_matches if len(addr) >= 26 and len(addr) <= 35]
    
    # Extract street addresses
    street_matches = re.findall(street_pattern, text, re.IGNORECASE)
    # Clean up street addresses by getting the full match
    for match in re.finditer(street_pattern, text, re.IGNORECASE):
        street_addresses.append(match.group(0).strip())
    
    # Remove duplicates while preserving order
    emails = list(dict.fromkeys(emails))
    ssns = list(dict.fromkeys(ssns))
    bitcoin_addresses = list(dict.fromkeys(bitcoin_addresses))
    street_addresses = list(dict.fromkeys(street_addresses))
    
    return (emails, ssns, bitcoin_addresses, street_addresses)

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


from collections import Counter

def tfidf_data(reviews_ser: pd.Series, review: str) -> pd.DataFrame:
    """
    Calculate TF-IDF metrics for words in a given review compared to a corpus of reviews.
    
    Args:
        reviews_ser: Series containing all reviews
        review: Single review to analyze
        
    Returns:
        DataFrame with columns for count, term frequency, inverse document frequency, and tf-idf
    """
    # Convert review to lowercase and split into words
    words = re.findall(r'\b\w+\b', review.lower())
    
    # Count word frequencies in the review
    word_counts = Counter(words)
    
    # Calculate term frequency (TF)
    total_words = len(words)
    tf = {word: count/total_words for word, count in word_counts.items()}
    
    # Calculate inverse document frequency (IDF)
    total_documents = len(reviews_ser)
    idf = {}
    
    for word in word_counts:
        # Count documents containing the word
        pattern = r'\b' + re.escape(word) + r'\b'
        docs_with_word = reviews_ser.str.lower().str.count(pattern).gt(0).sum()
        idf[word] = np.log(total_documents / docs_with_word)
    
    # Create DataFrame with all metrics
    result_df = pd.DataFrame({
        'cnt': pd.Series(word_counts),
        'tf': pd.Series(tf),
        'idf': pd.Series(idf)
    })
    
    # Calculate TF-IDF
    result_df['tfidf'] = result_df['tf'] * result_df['idf']
    
    return result_df

def relevant_word(tfidf_df: pd.DataFrame) -> str:
    """
    Find the word with the highest TF-IDF score.
    
    Args:
        tfidf_df: DataFrame containing TF-IDF metrics
        
    Returns:
        String containing the word with highest TF-IDF score
    """
    # Return word with highest TF-IDF score
    return tfidf_df['tfidf'].idxmax()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


from collections import Counter

def hashtag_list(tweets):
    """
    Extract hashtags from tweet texts and return them as lists without the '#' symbol.
    
    Parameters:
    tweets (pd.Series): Series of tweet texts
    
    Returns:
    pd.Series: Series where each element is a list of hashtags found in the corresponding tweet
    """
    def extract_hashtags(text):
        if not isinstance(text, str):
            return []
        
        # Find all hashtags using regex
        # Look for # followed by non-whitespace characters
        hashtags = re.findall(r'#\S+', text)
        
        # Remove the # symbol from each hashtag
        return [tag[1:] for tag in hashtags]
    
    return tweets.apply(extract_hashtags)

def most_common_hashtag(hashtag_lists):
    """
    Find the most common hashtag for each tweet based on overall frequency in the series.
    
    Parameters:
    hashtag_lists (pd.Series): Series where each element is a list of hashtags
    
    Returns:
    pd.Series: Series with the most common hashtag for each tweet (or NaN if no hashtags)
    """
    # Count all hashtags across the entire series
    all_hashtags = []
    for hashtag_list in hashtag_lists:
        all_hashtags.extend(hashtag_list)
    
    # Create frequency dictionary of all hashtags
    hashtag_counts = Counter(all_hashtags)
    
    def get_most_common(hashtags):
        if not hashtags:  # If empty list
            return pd.NA
        elif len(hashtags) == 1:  # If only one hashtag
            return hashtags[0]
        else:
            # Get the hashtag with highest overall frequency
            return max(hashtags, key=lambda x: hashtag_counts[x])
    
    return hashtag_lists.apply(get_most_common)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------




    



def count_hashtags(text):
    """Count number of hashtags in tweet."""
    return len(re.findall(r'#\w+', text))

def get_most_common_hashtag(text):
    """Get the most common hashtag in tweet."""
    hashtags = re.findall(r'#\w+', text.lower())
    if not hashtags:
        return ''
    return Counter(hashtags).most_common(1)[0][0]

def count_tags(text):
    """Count number of @ mentions (tags) in tweet."""
    # Match @ followed by at least one alphanumeric character
    return len(re.findall(r'@[a-zA-Z0-9]+', text))

def count_links(text):
    """Count number of hyperlinks in tweet."""
    return len(re.findall(r'https?://\S+', text))

def is_retweet(text):
    """Check if tweet starts with RT."""
    return text.strip().startswith('RT')

def clean_text(text):
    """Clean tweet text according to specified steps."""
    # Step 1: Replace meta-information with space
    text = re.sub(r'RT\s+', ' ', text)  # Remove retweet
    text = re.sub(r'@[a-zA-Z0-9]+', ' ', text)  # Remove tags
    text = re.sub(r'https?://\S+', ' ', text)  # Remove links
    text = re.sub(r'#\w+', ' ', text)  # Remove hashtags
    
    # Step 2: Replace non-alphanumeric (except spaces) with space
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Step 3: Convert to lowercase
    text = text.lower()
    
    # Step 4: Normalize spaces and strip
    text = ' '.join(text.split())
    
    return text

def create_features(ira):
    """
    Create feature DataFrame from IRA tweets.
    
    Parameters:
    -----------
    ira : pandas.DataFrame
        DataFrame with a single 'text' column containing tweets
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features extracted from tweets
    """
    features = pd.DataFrame(index=ira.index)
    
    # Extract all features
    features['text'] = ira['text'].apply(clean_text)
    features['num_hashtags'] = ira['text'].apply(count_hashtags)
    features['mc_hashtags'] = ira['text'].apply(get_most_common_hashtag)
    features['num_tags'] = ira['text'].apply(count_tags)
    features['num_links'] = ira['text'].apply(count_links)
    features['is_retweet'] = ira['text'].apply(is_retweet)
    
    # Reorder columns as specified
    return features[['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']]