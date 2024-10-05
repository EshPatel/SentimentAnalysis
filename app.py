from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scrape_reddit import reddit_scrape
from scrape_twitter import twitter_scrape
from scrape_youtube import youtube_scrape
from scrape_instagram import instagram_scrape

# Initialize FastAPI
app = FastAPI()

# Request body model for video_id and search_query
class YouTubeRequest(BaseModel):
    video_id: str

class InstagramRequest(BaseModel):
    search_query: str

class TwitterRequest(BaseModel):
    search_query: str

class RedditRequest(BaseModel):
    search_query: str

# YouTube
@app.post('/scrape/youtube')
async def scrape_youtube(request: YouTubeRequest):
    video_id = request.video_id
    if not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")

    result = youtube_scrape(video_id)
    return {"result": result}

# Instagram
@app.post('/scrape/instagram')
def scrape_instagram(request: InstagramRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Instagram search query")

    result = instagram_scrape(search_query) 
    return {"message": result}

# Twitter 
@app.post('/scrape/twitter')
async def scrape_twitter(request: TwitterRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Twitter search query")

    result = await twitter_scrape(search_query) 
    return {"message": result}

# Reddit 
@app.post('/scrape/reddit')
async def scrape_reddit(request: RedditRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Reddit search query")

    result = reddit_scrape(search_query)
    return {"message": result}

# Run FastAPI app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
