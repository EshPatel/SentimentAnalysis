import os
import praw
import csv
from configparser import ConfigParser
from datetime import datetime

def convert_unix_to_date(timestamp):
    # Convert the timestamp to a datetime object
    date_obj = datetime.utcfromtimestamp(timestamp)
    
    # Format the datetime object as a string
    formatted_date = date_obj.strftime('%a %b %d %H:%M:%S +0000 %Y')
    
    return formatted_date

def reddit_scrape(search_query):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/reddit_config.ini'))
    client_id = config['RDT']['client_id']
    client_secret = config['RDT']['client_secret']
    user_agent = config['RDT']['user_agent']
    username = config['RDT']['username'] 
    password = config['RDT']['password']  
    post_limit = config['RDT']['post_limit'] 
    limit = int(post_limit)

    # login reddit
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password
    )

    # setting query and get realted posts with specific limit
    subreddit = reddit.subreddit(search_query)
    new_posts = subreddit.new(limit=limit)

    # saving scraped posts data to csv file
    with open(os.path.join(current_dir, 'csv_outputs/reddit_posts.csv'), mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Post Number', 'Platform', 'Username', 'Content URL', 'Text', 'Creation Date', 'Likes', 'Comments', 'Additional Info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, post in enumerate(new_posts):
            writer.writerow({
                "Post Number": i + 1,
                "Platform": "Reddit",
                "Username": post.author,
                "Content URL": post.url,
                "Text": post.title,
                "Creation Date": convert_unix_to_date(post.created_utc),
                "Likes": post.score,
                "Comments": post.num_comments,
                "Additional Info": f"ID: {post.id}"
            })

    return "Reddit data scraped and saved to CSV"
