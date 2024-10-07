## **Social Media Scraper**
This project is a set of social media scrapers for platforms like YouTube, Twitter, Instagram, and Reddit, built using FastAPI. The project follows the abstract factory design pattern and uses asynchronous functions for efficient scraping.


## Prerequisites
Python 3.8 or higher
Postman (optional for testing)


## Installation
Clone the repository:
git clone https://github.com/eaysu/social-media-scraper.git
cd social-media-scraper

## Create a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

## Install the dependencies:
pip install -r requirements.txt


## Running the Application
Start the FastAPI server using uvicorn:
uvicorn main:app --reload


## Postman Usage
Reddit: 
POST http://127.0.0.1:8000/scrape/reddit
{
  "search_query": "topic_name" (ex. turkiye)
}

Youtube: 
POST http://127.0.0.1:8000/scrape/youtube
{
  "video_id": "video_id" (ex. S4_bc5hR6OM)
}

Twitter: 
POST http://127.0.0.1:8000/scrape/twitter
{
  "search_query": "query" (ex. from:haskologlu)
}

Instagram: 
POST http://127.0.0.1:8000/scrape/instagram
{
  "search_query": "username" (ex. allianzturkiye)
}


