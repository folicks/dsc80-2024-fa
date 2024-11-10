
# Do NOT edit this file. Instead, just call it from the command line,
# using the instructions in the assignment notebook.

import sys
questions = sys.argv[1:]


valid_ids = ['q1', 'q2', 'q3', 'q4', 'q5']
break_flag = False
invalid_ids = []
for question in questions:
    if question != 'all' and question not in valid_ids:
        invalid_ids.append(question)

if len(invalid_ids) > 0:
    print(str(invalid_ids) + ' is/are not a valid question number(s). The possible question numbers are ' + str(valid_ids) + '.')
    sys.exit()

# Initialize Otter
import otter
grader = otter.Notebook("lab.ipynb")

# %load_ext autoreload
# %autoreload 2

from lab import *

import pandas as pd
import numpy as np
import os
import re

if 'q1' in questions or questions == [] or 'all' in questions:
    print(grader.check("q1"))

# experiment with extract_personal using the file s below
fp = os.path.join('data', 'messy.txt')
s = open(fp, encoding='utf8').read()

# don't change this cell, but do run it -- it is needed for the tests
test_fp = os.path.join('data', 'messy.test.txt')
test_s = open(test_fp, encoding='utf8').read()
emails, ssn, bitcoin, addresses = extract_personal(test_s)

if 'q2' in questions or questions == [] or 'all' in questions:
    print(grader.check("q2"))

# experiment with tfidf_data using reviews_ser and review below 
fp = os.path.join('data', 'reviews.txt')
reviews_ser = pd.read_csv(fp, header=None).squeeze("columns")
review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()

# don't change this cell, but do run it -- it is needed for the tests
fp = os.path.join('data', 'reviews.txt')
reviews_ser = pd.read_csv(fp, header=None).squeeze("columns")
review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
q3_tfidf = tfidf_data(reviews_ser, review)

try:
    q3_rel = relevant_word(q3_tfidf)
except:
    q3_rel = None

if 'q3' in questions or questions == [] or 'all' in questions:
    print(grader.check("q3"))

# The public tests don't test your work on the `ira` data,
# but the hidden tests do.
# So, make sure to thoroughly test your work yourself!
fp = os.path.join('data', 'ira.csv')
ira = pd.read_csv(fp, names=['id', 'name', 'date', 'text'])

if 'q4' in questions or questions == [] or 'all' in questions:
    print(grader.check("q4"))

# The doctests/public tests don't test your work on the `ira` data,
# but the hidden tests do.
# So, make sure to thoroughly test your work yourself!
fp = os.path.join('data', 'ira.csv')
ira = pd.read_csv(fp, names=['id', 'name', 'date', 'text'])

# don't change this cell, but do run it -- it is needed for the tests
# (yes, we know it says "hidden" – there are still truly hidden tests in this question)
fp_hidden = 'data/ira_test.csv'
ira_hidden = pd.read_csv(fp_hidden, header=None)
text_hidden = ira_hidden.iloc[:, -1:]
text_hidden.columns = ['text']

test_hidden = create_features(text_hidden)

if 'q5' in questions or questions == [] or 'all' in questions:
    print(grader.check("q5"))


