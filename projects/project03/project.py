# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------



def get_robot_delay(url="https://www.gutenberg.org/robots.txt", default_delay=0.5):
    """
    Fetches the delay specified in Project Gutenberg's robots.txt.
    If no delay is specified, return a default delay.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Look for a line with the Crawl-delay directive
        match = re.search(r"Crawl-delay: (\d+)", response.text)
        if match:
            return float(match.group(1))
    except requests.RequestException:
        pass
    return default_delay

# Store the last request time to implement the delay between requests.
last_request_time = None

def get_book(url):
    global last_request_time
    
    # Ensure we respect the delay between requests.
    delay = get_robot_delay()
    if last_request_time is not None:
        time_since_last_request = time.time() - last_request_time
        if time_since_last_request < delay:
            time.sleep(delay - time_since_last_request)
    
    # Fetch the book content.
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    
    # Update the last request time.
    last_request_time = time.time()
    
    # Find the start and end markers.
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    
    if start_match and end_match:
        # Extract content between START and END markers, and replace \r\n with \n.
        content = text[start_match.end():end_match.start()]
        return content.replace('\r\n', '\n')
    else:
        raise ValueError("Could not find start or end markers in the book text.")



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    # Define the START and STOP tokens.
    START_TOKEN = '\x02'
    STOP_TOKEN = '\x03'
    
    # Regular expression to split by paragraphs (two or more newlines).
    paragraphs = re.split(r'\n{2,}', book_string.strip())
    
    # Regular expression to find tokens: words, numbers, punctuation (not underscore).
    token_pattern = re.compile(r"\w+|[^\w\s]")

    # Tokenize each paragraph, wrapping in START and STOP tokens.
    tokens = [START_TOKEN]
    for paragraph in paragraphs:
        # Find all tokens in the paragraph.
        tokens += token_pattern.findall(paragraph)
        tokens.append(STOP_TOKEN)
    
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM:
    def __init__(self, corpus):
        # Train the language model on initialization
        self.train(corpus)
    
    def train(self, tokens):
        # Get the unique tokens and count them
        unique_tokens = set(tokens)
        total_unique_tokens = len(unique_tokens)
        
        # Calculate uniform probability for each unique token
        uniform_probability = 1 / total_unique_tokens
        
        # Create a Series where each unique token has the same probability
        self.mdl = pd.Series({token: uniform_probability for token in unique_tokens})
    
    def probability(self, token_sequence):
        # Compute the probability of the sequence by multiplying individual token probabilities
        prob = 1.0
        for token in token_sequence:
            if token in self.mdl:
                prob *= self.mdl[token]
            else:
                return 0  # If any token is not in the model, return 0
        return prob
    
    def sample(self, M):
        # Generate M tokens by sampling with replacement from the unique tokens
        tokens = np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values)
        return ' '.join(tokens)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

from collections import Counter

class UnigramLM:
    def __init__(self, corpus):
        # Train the language model on initialization
        self.train(corpus)
    
    def train(self, tokens):
        # Count total number of tokens and their frequencies
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        
        # Calculate empirical probabilities and store them in a Series
        self.mdl = pd.Series({token: count / total_tokens for token, count in token_counts.items()})
    
    def probability(self, token_sequence):
        # Compute the probability of the sequence by multiplying individual token probabilities
        prob = 1.0
        for token in token_sequence:
            if token in self.mdl:
                prob *= self.mdl[token]
            else:
                return 0  # If any token is not in the model, return 0
        return prob
    
    def sample(self, M):
        # Generate M tokens by sampling with replacement from the unique tokens
        tokens = np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values)
        return ' '.join(tokens)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


import numpy as np
import pandas as pd
from collections import Counter

class NGramLM:
    def __init__(self, N, tokens):
        self.N = N
        self.tokens = tokens
        self.prev_mdl = None if N == 1 else NGramLM(N - 1, tokens)
        self.mdl = None  # This will hold the DataFrame after training
        self.train(tokens)
    
    def create_ngrams(self, tokens):
        """Generates a list of N-Grams from a list of tokens."""
        return [tuple(tokens[i:i + self.N]) for i in range(len(tokens) - self.N + 1)]

    def train(self, tokens):
        """Trains the N-Gram model by counting occurrences of each N-Gram and (N-1)-Gram."""
        ngrams = self.create_ngrams(tokens)
        n1grams = [ngram[:-1] for ngram in ngrams]
        ngram_counts = Counter(ngrams)
        n1gram_counts = Counter(n1grams)

        self.mdl = pd.DataFrame({
            'ngram': list(ngram_counts.keys()),
            'n1gram': [ngram[:-1] for ngram in ngram_counts.keys()],
            'prob': [ngram_counts[ng] / n1gram_counts[ng[:-1]] for ng in ngram_counts.keys()]
        })
    def get_probability(self, ngram):
        """
        Retrieves the probability of an N-Gram from the model. If it doesn't exist,
        recursively checks lower-order models until a probability is found or returns 0.
        """
        if len(ngram) == self.N:
            row = self.mdl[self.mdl['ngram'] == ngram]
            if not row.empty:
                return row['prob'].values[0]
        
        # If this N-Gram does not exist in current model, check lower-order models
        if self.prev_mdl:
            return self.prev_mdl.get_probability(ngram[:-1])
        
        # If no match is found in any model (unigram case), return 0
        return 0

    def probability(self, sentence):
        """
        Calculates the probability of a sentence by multiplying the probabilities
        of each N-Gram in the sentence.
        """
        total_prob = 1.0
        for i in range(len(sentence) - self.N + 1):
            ngram = tuple(sentence[i:i + self.N])
            prob = self.get_probability(ngram)
            if prob == 0:
                return 0  # Return 0 if any N-Gram has 0 probability
            total_prob *= prob
        
        return total_prob

    def sample_next_token(self, context):
        """
        Samples the next token based on the context (N-1)-Gram.
        If no tokens are found, returns '\x03'.
        """
        # Filter mdl to find rows matching the context
        options = self.mdl[self.mdl['n1gram'] == context]
        if options.empty:
            return '\x03'
        
        # Sample based on probabilities
        tokens = options['ngram'].apply(lambda x: x[-1]).values
        probabilities = options['prob'].values
        return np.random.choice(tokens, p=probabilities)

    def sample(self, M):
        """
        Generates a sample sentence of M tokens using the N-Gram model.
        Starts with '\x02' and ends with '\x03' when the length reaches M tokens.
        """
        sentence = ['\x02']
        
        while len(sentence) - 1 < M:
            # Get the (N-1) most recent tokens as context
            context = tuple(sentence[-(self.N - 1):])
            
            # Sample the next token based on the current context
            next_token = self.sample_next_token(context)
            sentence.append(next_token)
            
            # If we reach the end-of-sequence token, break
            if next_token == '\x03':
                break

        return ' '.join(sentence)


