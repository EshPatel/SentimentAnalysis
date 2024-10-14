import os
import re
import csv
import requests
from time import sleep
from random import randint
from bs4 import BeautifulSoup
from configparser import ConfigParser

# the function that converts the input the proper url format
def format_for_url(text):
    # mapping of turkish characters to english equivalents
    turkish_char_map = {
        'ç': 'c', 'Ç': 'C', 'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I', 'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S', 'ü': 'u', 'Ü': 'U'
    }

    # Replace Turkish characters
    for turkish_char, english_char in turkish_char_map.items():
        text = text.replace(turkish_char, english_char)
    # Replace spaces with hyphens and remove non-URL-friendly characters
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[^a-zA-Z0-9\-]', '', text)
    return text.lower()

def sikayetvar_scrape(search_query):
    # set up csv file for writing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(current_dir, 'csv_outputs/sikayetvar_data.csv')

    config = ConfigParser()
    config.read(os.path.join(current_dir, 'config_files/sikayetvar_config.ini'))
    complaint_limit = config['SV']['complaint_limit']
    complaint_limit = int(complaint_limit)

    # define a valid User-Agent header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    page_num = 1

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Post Number', 'Platform', 'Username', 'Content URL', 'Text', 'Creation Date', 'Additional Info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # initialize variables 
        search_query = format_for_url(search_query)   
        url = f"https://www.sikayetvar.com/{search_query}/"

        complaint_count = 0

        while complaint_limit > complaint_count:
            # update url for pagination
            if page_num >= 2:
                # Sleep to avoid blocks
                sleep_time = randint(5, 10)
                print(f"[[ Program sleeps for {sleep_time} seconds to avoid blocks ]]")
                sleep(sleep_time)
                url = f"https://www.sikayetvar.com/arcelik?page={page_num}"

            try:
                # request main page
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    response = BeautifulSoup(response.text, "html.parser")  

                    # find all article sections that contain complaint headers
                    articles = response.find_all("article", class_="card-v2 ga-v ga-c")  

                    page_num += 1

                    for article in articles:
                        complaint_count += 1
                        if complaint_count > complaint_limit:
                            break

                        # extract the complaint header/title
                        complaint_header = article.find("h2", class_="complaint-title")
                        if complaint_header:
                            header_text = complaint_header.get_text(strip=True)

                            # format header for URL
                            formatted_header = format_for_url(header_text)
                            complaint_url = f"https://www.sikayetvar.com/arcelik/{formatted_header}"
                            
                            try:
                                # Request the individual complaint page
                                complaint_response = requests.get(complaint_url, headers=headers)
                                if complaint_response.status_code == 200:
                                    complaint_soup = BeautifulSoup(complaint_response.text, "html.parser")
                                    
                                    # extract profile name
                                    profile_name = complaint_soup.find("span", class_="username")
                                    profile_name_text = profile_name.get_text(strip=True) if profile_name else "N/A"
                                    
                                    # extract post time
                                    post_time = complaint_soup.find("div", class_="js-tooltip time")
                                    post_time_text = post_time.get_text(strip=True) if post_time else "N/A"
                                    
                                    # extract complaint text
                                    text_div = complaint_soup.find("div", class_="complaint-detail-description")
                                    complaint_text = text_div.get_text(separator=" ", strip=True) if text_div else "N/A"

                                    # Write to CSV
                                    writer.writerow({
                                        "Post Number": complaint_count,
                                        "Platform": "Sikayet Var",
                                        "Username": profile_name_text,
                                        "Content URL": complaint_url,
                                        "Text": complaint_text,
                                        "Creation Date": post_time_text,
                                        "Additional Info": "N/A"
                                    })
                                    print(f"Added complaint #{complaint_count}: {profile_name_text} - {post_time_text}")
         
                            except Exception as e:
                                print(f"Error fetching complaint page: {complaint_url} - {e}")
            except Exception as e:
                print(f"Error fetching main page {url}: {e}")        

    return "Sikayet Var data scraped and saved to CSV" 