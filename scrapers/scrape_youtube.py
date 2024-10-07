import os
import csv
from googleapiclient.discovery import build
from configparser import ConfigParser

def youtube_scrape(keyword):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/youtube_config.ini'))
    api_key = config['YT']['api_key']
    comment_limit = int(config['YT']['comment_limit'])
    video_limit = int(config['YT']['video_limit'])

    # login youtube
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Search for videos related to the keyword
    search_request = youtube.search().list(
        part='snippet',
        q=keyword,
        type='video',
        maxResults=video_limit
    )
    search_response = search_request.execute()

    # Open the CSV file
    with open(os.path.join(current_dir, 'csv_outputs/youtube_data.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Name', 'Creator', 'Comment Number', 'Comment Author', 'Comment Text'])
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_name = item['snippet']['title']
            channel_name = item['snippet']['channelTitle']
            
            # Request comments for each video
            comment_request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=comment_limit
            )
            comment_response = comment_request.execute()

            for i, comment_item in enumerate(comment_response['items']):
                comment = comment_item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = comment_item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                writer.writerow([video_name, channel_name, i+1, author, comment])

    return "YouTube data scraped and saved to CSV."

# Usage
# youtube_scrape('your_keyword_here')
