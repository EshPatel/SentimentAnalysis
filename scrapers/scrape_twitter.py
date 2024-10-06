import os
import asyncio
import csv
from twikit import Client, TooManyRequests
import time
from datetime import datetime
from configparser import ConfigParser
from random import randint

async def get_tweets(tweets, search_query, client):
    # fetches tweets with a delay
    if tweets is None:
        # get tweets
        print(f"{datetime.now()} - Getting tweets...")
        tweets = await client.search_tweet(search_query, product='Top')
    else:
        # to prevent from banning, it sleeps 5 to 10 seconds in every specific tweet count (likely 20 tweets)
        wait_time = randint(5, 10)
        print(f"{datetime.now()} - Getting next tweets after waiting {wait_time} seconds...")
        await asyncio.sleep(wait_time)
        tweets = await tweets.next()

    return tweets

async def scrape_tweets(client, search_query, tweet_limit, current_dir):
    tweet_count = 0
    tweets = None

    # scraping the specified number of tweets
    while tweet_count < tweet_limit:
        try:
            tweets = await get_tweets(tweets, search_query, client)
        except TooManyRequests as e:
            # is scraping operation come across rate limit it sleeps for a while to prevent from banning
            rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
            print(f"{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}")
            wait_time = rate_limit_reset - datetime.now()
            await asyncio.sleep(wait_time.total_seconds())
            continue

        # if program couldnt find tweet or more tweets according to query it returns message
        if not tweets:
            print(f"{datetime.now()} - No more tweets found.")
            break

        # creating csv file and saving tweet data to csv file
        for tweet in tweets:
            tweet_count += 1
            
            with open(os.path.join(current_dir, 'csv_outputs/tweets.csv'), mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Tweet Count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                writer.writerow({
                    "Tweet Count": tweet_count,
                    "Username": tweet.user.name,
                    "Text": tweet.text,
                    "Created At": tweet.created_at,
                    "Retweets": tweet.retweet_count,
                    "Likes": tweet.favorite_count
                })

        print(f"{datetime.now()} - Got {tweet_count} tweets!") 
    print(f"{datetime.now()} - Done! Got {tweet_count} tweets and replies.")

async def twitter_scrape(search_query):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/twitter_config.ini'))
    username = config['X']['username']
    email = config['X']['email']
    password = config['X']['password']
    tweet_limit = config['X']['tweet_limit']
    limit = int(tweet_limit)

    # login twitter
    client = Client(language='tur')
    await client.login(auth_info_1=username, auth_info_2=email, password=password)
    client.save_cookies(os.path.join(current_dir, 'cookies.json'))
    client.load_cookies(os.path.join(current_dir, 'cookies.json'))

    # scrape tweets
    await scrape_tweets(client, search_query, limit, current_dir)

    return "Twitter data scraped and saved to CSV"
