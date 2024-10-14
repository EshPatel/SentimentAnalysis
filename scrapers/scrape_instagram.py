import os
import instaloader
from datetime import datetime
import time
from random import randint, choice
import csv
from configparser import ConfigParser

MAX_POSTS = 5
MIN_COMMENTS = 20
COMMENTS_PER_BATCH = 30
POST_PER_BATCH = 10

def get_comments(post, comment_limit):
    # get post comments and write on created csv file
    with open('social_media_scraper/instagram_comments.csv', mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Count', 'Username', 'Comment', 'Comment Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # iterate over comments in batches
        for comment_count, comment in enumerate(post.get_comments()):
            if comment_count >= comment_limit:
                break

            comment_count += 1

            # write the row with comment details
            writer.writerow({
                "Count": comment_count,
                # "Username": comment.owner.username,
                "Comment": comment.text.replace('\n', ' ') if comment.text else 'No text', 
                # "Comment Date": comment.created_at_utc.strftime('%Y-%m-%d %H:%M:%S'),
            })

def get_posts_and_comments(profile_name, loader, post_limit, comment_limit, current_dir):
    # fetch posts and their comments
    print(f"{datetime.now()} - Getting posts from {profile_name}...")

    # load profile
    profile = instaloader.Profile.from_username(loader.context, profile_name)

    post_count = 0

    # get post data and write on created csv file
    with open(os.path.join(current_dir, 'csv_outputs/instagram_posts.csv'), mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Post Number', 'Platform', 'Username', 'Content URL', 'Text', 'Creation Date', 'Additional Info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        post_count = 0 

        # loop over posts
        for post in profile.get_posts():
            if post_count >= post_limit:
                break

            post_count += 1

            if post_count % 20 == 0:
                sleep_time = randint(10, 20)
                time.sleep(sleep_time)
                print(f"To prevent rate limit program waits {sleep_time} seconds")

            # write the row with post details
            # Write the row with standardized column names
            writer.writerow({
                "Post Number": post_count,
                "Platform": "Instagram",
                "Username": profile.username,
                "Content URL": f'https://www.instagram.com/p/{post.shortcode}/',
                "Text": post.caption.replace('\n', ' ') if post.caption else 'No caption',
                "Creation Date": post.date_utc.strftime('%Y-%m-%d %H:%M:%S'),
                #"Likes": post.likes,
                #"Comments": post.comments,
                "Additional Info": "N/A"
            })

            # Optionally, fetch and write comments separately
            # get_comments(post, comment_limit)


def instagram_scrape(search_query):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/instagram_config.ini'))
    username = config['IG']['username']
    password = config['IG']['password']
    post_limit = config['IG']['post_limit']
    comment_limit = config['IG']['comment_limit']
    p_limit = int(post_limit)
    c_limit = int(comment_limit)

    # login instagram
    loader = instaloader.Instaloader()
    if os.path.exists(os.path.join(current_dir, f'{username}_session')):
        loader.load_session_from_file(username)
        print("Previous session loaded succesfully")
    else:
        loader.login(username, password)
        print("Login succesfull")
        loader.save_session_to_file(os.path.join(current_dir, f'{username}_session'))
        print("login sucessfull and session file saved")
    
    try:
        get_posts_and_comments(search_query, loader, p_limit, c_limit, current_dir)
    except (instaloader.exceptions.TooManyRequestsException,
            instaloader.exceptions.QueryReturnedNotFoundException,
            instaloader.exceptions.QueryReturnedBadRequestException,
            instaloader.exceptions.ConnectionException,
            instaloader.exceptions.LoginException,
            instaloader.exceptions.LoginRequiredException) as e:
        print(e)

    return "Instagram data scraped and saved to CSV" 



