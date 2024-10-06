import os
import csv
from googleapiclient.discovery import build
from configparser import ConfigParser

def youtube_scrape(video_id):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/youtube_config.ini'))
    api_key = config['YT']['api_key']
    comment_limit = config['YT']['comment_limit']
    limit = int(comment_limit)

    # login youtube
    youtube = build('youtube', 'v3', developerKey=api_key)

    # scrape data from specific video
    video_request = youtube.videos().list(part='snippet', id=video_id)
    video_response = video_request.execute()

    # seperate scraped data to video name, description and channel name
    video_name = video_response['items'][0]['snippet']['title']
    video_description = video_response['items'][0]['snippet']['description']
    channel_name = video_response['items'][0]['snippet']['channelTitle']

    # create csv file, append video data, scrape comments and also save these comments to csv file
    with open(os.path.join(current_dir, 'csv_outputs/youtube_data.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        writer.writerow(['Video Name', video_name])
        writer.writerow(['Description', video_description])
        writer.writerow(['Creator', channel_name])
        writer.writerow([]) 

        writer.writerow(['Comment Number', 'Comment Author', 'Comment Text'])

        comment_request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=limit
        )
        comment_response = comment_request.execute()

        for i, item in enumerate(comment_response['items']):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            writer.writerow([i+1, author, comment])

    return "Youtube data scraped and saved to CSV"
