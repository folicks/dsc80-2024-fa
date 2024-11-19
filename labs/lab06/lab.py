# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(page_content):
    soup = bs4.BeautifulSoup(page_content, 'lxml')
    book_links = []

    # Map star rating text to numbers
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    # Loop through each book on the page
    for article in soup.find_all("article", class_="product_pod"):
        # Check rating
        rating_class = article.find("p", class_="star-rating")["class"][1]
        rating = rating_map.get(rating_class, 0)
        
        # Check price
        price_text = article.find("p", class_="price_color").text
        price = float(price_text[2:])  # Remove the £ symbol and convert to float
        
        if rating >= 4 and price < 50:
            # Extract the URL and strip the protocol
            link = article.find("h3").a["href"]
            book_links.append(link.lstrip("/"))
    
    return book_links

def get_product_info(book_page_content, categories):
    soup = bs4.BeautifulSoup(book_page_content, 'lxml')
    
    # Extract category
    category = soup.find("ul", class_="breadcrumb").find_all("a")[-1].text.strip()
    
    # Return None if the category is not in the target categories
    if category not in categories:
        return None

    # Extract information from the product table
    table = soup.find("table", class_="table table-striped")
    product_info = {row.th.text: row.td.text for row in table.find_all("tr")}

    # Extract specific fields
    upc = product_info.get("UPC", "")
    product_type = product_info.get("Product Type", "")
    price_excl_tax = product_info.get("Price (excl. tax)", "")
    price_incl_tax = product_info.get("Price (incl. tax)", "")
    tax = product_info.get("Tax", "")
    availability = product_info.get("Availability", "")
    number_of_reviews = product_info.get("Number of reviews", "0")
    
    # Extract description, if available
    description = soup.find("meta", attrs={"name": "description"})
    description_text = description["content"].strip() if description else ""
    
    # Extract title
    title = soup.find("h1").text.strip()

    # Extract rating
    rating_class = soup.find("p", class_="star-rating")["class"][1]

    # Construct the dictionary
    book_info = {
        "UPC": upc,
        "Product Type": product_type,
        "Price (excl. tax)": price_excl_tax,
        "Price (incl. tax)": price_incl_tax,
        "Tax": tax,
        "Availability": availability,
        "Number of reviews": number_of_reviews,
        "Category": category,
        "Rating": rating_class,
        "Description": description_text,
        "Title": title
    }

    return book_info

def scrape_books(k, categories):
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"
    books_data = []

    for page in range(1, k + 1):
        # Get page content
        page_url = base_url.format(page)
        response = requests.get(page_url)
        
        # If the page is not found, break out of the loop
        if response.status_code != 200:
            break

        # Extract book links from the current page
        book_links = extract_book_links(response.text)

        for link in book_links:
            # Get the full URL for each book
            book_url = "http://books.toscrape.com/catalogue/" + link
            book_response = requests.get(book_url)
            
            # Get book information
            book_info = get_product_info(book_response.text, categories)
            
            # Only append if book_info is not None (it met category criteria)
            if book_info:
                books_data.append(book_info)

    # Convert to DataFrame
    df = pd.DataFrame(books_data)
    return df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    """
    Fetches the historical stock data for a given ticker symbol, year, and month.
    
    Parameters:
    ticker (str): The stock ticker symbol.
    year (int): The year of the data to fetch.
    month (int): The month of the data to fetch.
    
    Returns:
    pd.DataFrame: A DataFrame containing stock data for the specified month.
    """
    # Define the API key and dates for the range
    api_key = "TEj6e6njs900yG5LGYcwWxdKNdpE8oN0"  # Replace with your actual API key
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month + 1:02d}-01" if month < 12 else f"{year + 1}-01-01"
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}&from={start_date}&to={end_date}'
    
    # Make the API request and retrieve data as JSON
    response = requests.get(url)
    data = response.json()
    
    # Check if 'historical' data is available in the response
    if 'historical' not in data:
        print("No historical data found for this ticker and date range.")
        return pd.DataFrame()  # Return empty DataFrame if no data found
    
    # Convert the 'historical' data to a DataFrame without filtering
    df = pd.DataFrame(data['historical'])
    
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to include only rows that match the specified year and month
    df = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]
    
    return df

def stock_stats(df):
    """
    Calculates the percent change and total transaction volume for the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing stock data for a specific month.
    
    Returns:
    tuple: A tuple containing (percent change, total transaction volume) as strings.
    """
    # Ensure the DataFrame has the necessary data
    if df.empty or 'open' not in df.columns or 'close' not in df.columns:
        return ("N/A", "N/A")
    
    # Sort the DataFrame by date to get the first and last records
    df = df.sort_values(by='date')
    
    # Calculate percent change
    start_price = df.iloc[0]['open']
    end_price = df.iloc[-1]['close']
    percent_change = ((end_price - start_price) / start_price) * 100
    percent_change_str = f"{percent_change:+.2f}%"
    
    # Calculate total transaction volume
    total_volume = 0
    for _, row in df.iterrows():
        avg_price = (row['high'] + row['low']) / 2  # Midpoint of high and low
        volume = row['volume']
        total_volume += avg_price * volume  # Transaction volume for the day
    
    # Convert to billions and format as a string
    total_volume_billion = total_volume / 1e9
    total_volume_str = f"{total_volume_billion:.2f}B"
    
    return percent_change_str, total_volume_str


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


from typing import List, Dict

def get_comment_data(comment_id: int) -> Dict:
    """
    Fetch comment data from Hacker News API for a given comment ID.
    
    Args:
        comment_id (int): The ID of the comment to fetch
        
    Returns:
        dict: Comment data from the API
    """
    url = f'https://hacker-news.firebaseio.com/v0/item/{comment_id}.json'
    response = requests.get(url)
    return response.json()

def traverse_comments(comment_id: int, comments_list: List[Dict]) -> None:
    """
    Recursively traverse comment tree using DFS and collect all valid comments.
    
    Args:
        comment_id (int): Current comment ID to process
        comments_list (list): List to store valid comments
    """
    comment_data = get_comment_data(comment_id)
    
    # Skip if comment is None or dead
    if comment_data is None or comment_data.get('dead', False):
        return
        
    # If it's a valid comment (not the story itself), add it to our list
    if comment_data.get('type') == 'comment':
        comments_list.append({
            'id': comment_data.get('id'),
            'by': comment_data.get('by'),
            'text': comment_data.get('text'),
            'parent': comment_data.get('parent'),
            'time': pd.Timestamp(comment_data.get('time', 0), unit='s')
        })
    
    # Recursively process child comments in order
    for kid_id in comment_data.get('kids', []):
        traverse_comments(kid_id, comments_list)

def get_comments(storyid: int) -> pd.DataFrame:
    """
    Fetch all comments for a given Hacker News story and return them as a DataFrame.
    
    Args:
        storyid (int): The ID of the Hacker News story
        
    Returns:
        pd.DataFrame: DataFrame containing all comments, ordered by DFS traversal
    """
    # First, get the story data
    story_data = get_comment_data(storyid)
    
    # Check if story exists and has comments
    if not story_data or 'kids' not in story_data:
        return pd.DataFrame(columns=['id', 'by', 'text', 'parent', 'time'])
    
    # List to store all comments
    comments_list = []
    
    # Process each top-level comment in order
    for comment_id in story_data['kids']:
        traverse_comments(comment_id, comments_list)
    
    # Create DataFrame from collected comments
    df = pd.DataFrame(comments_list)
    
    # Verify number of comments matches story's descendant count
    # (excluding dead comments)
    expected_count = story_data.get('descendants', 0)
    if len(df) > expected_count:
        # Trim excess comments if necessary
        df = df.head(expected_count)
    
    return df[['id', 'by', 'text', 'parent', 'time']]
