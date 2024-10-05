import praw
import csv
from configparser import ConfigParser

def reddit_scrape(search_query):
    # getting credentials from config file
    config = ConfigParser()
    config.read('social_media_scraper/reddit_config.ini')
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
    with open('social_media_scraper/reddit_posts.csv', mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Count', 'Title', 'ID', 'Author', 'URL', 'Score', 'Comment count', 'Created']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, post in enumerate(new_posts):
            writer.writerow({
                "Count": i + 1,
                "Title": post.title,
                "ID": post.id,
                "Author": post.author,
                "URL": post.url,
                "Score": post.score,
                "Comment count": post.num_comments,
                "Created": post.created_utc
            })

    return "Reddit data scraped and saved to CSV"
