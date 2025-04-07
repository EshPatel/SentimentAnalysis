import os
import csv
import pandas as pd
from googleapiclient.discovery import build
from configparser import ConfigParser
from googleapiclient.errors import HttpError

def youtube_scrape(keyword, video_limit = None):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # getting credentials from config file
    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/youtube_config.ini'))
    api_key = config['YT']['api_key']
    comment_limit = int(config['YT']['comment_limit'])
    video_limit = video_limit if video_limit else int(config['YT']['video_limit'])

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
    
    all_data = []

    # Open the CSV file using DictWriter
    with open(os.path.join(current_dir, 'csv_outputs/youtube_data.csv'), mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Post Number', 'Platform', 'Username', 'Content URL', 'Text', 'Creation Date', 'Likes', 'Comments', 'Additional Info']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_name = item['snippet']['title']
            channel_name = item['snippet']['channelTitle']
            
            try:
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
                    # print(comment_item['snippet']['topLevelComment']['snippet'])

                    # Use writer.writerow with a dictionary
                    all_data.append({
                        "Post Number": i + 1,
                        "Platform": "YouTube",
                        "Username": author,
                        "Content URL": f'https://www.youtube.com/watch?v={video_id}',
                        "Text": comment,
                        "Creation Date": 'N/A',  # You can fetch the date if needed
                        "Likes": 'N/A',  # You can fetch the likes if needed
                        "Comments": 'N/A',  # You can fetch the number of comments if needed
                        "Additional Info": f"Video: {video_name}, Channel: {channel_name}"
                    })
            
            except HttpError as e:
                # Check for the error status that corresponds to "comments disabled"
                if e.resp.status == 403:
                    print(f"Comments disabled for video: {video_name} (ID: {video_id}), skipping...")
                else:
                    print(f"An error occurred: {e}")
                continue

    df = pd.DataFrame(all_data)
    return df

