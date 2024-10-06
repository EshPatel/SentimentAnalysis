from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from factories.youtube_factory import YouTubeScraperFactory
from factories.twitter_factory import TwitterScraperFactory
from factories.instagram_factory import InstagramScraperFactory
from factories.reddit_factory import RedditScraperFactory

app = FastAPI()

# Request body models
class YouTubeRequest(BaseModel):
    video_id: str

class InstagramRequest(BaseModel):
    search_query: str

class TwitterRequest(BaseModel):
    search_query: str

class RedditRequest(BaseModel):
    search_query: str

# Routes
@app.post('/scrape/youtube')
async def scrape_youtube(request: YouTubeRequest):
    video_id = request.video_id
    if not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")

    factory = YouTubeScraperFactory()
    scraper = factory.create_scraper()
    result = scraper.scrape(video_id)
    return {"result": result}

@app.post('/scrape/instagram')
def scrape_instagram(request: InstagramRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Instagram search query")

    factory = InstagramScraperFactory()
    scraper = factory.create_scraper()
    result = scraper.scrape(search_query)
    return {"message": result}

@app.post('/scrape/twitter')
async def scrape_twitter(request: TwitterRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Twitter search query")

    factory = TwitterScraperFactory()
    scraper = factory.create_scraper()
    result = await scraper.scrape(search_query)
    return {"message": result}

@app.post('/scrape/reddit')
async def scrape_reddit(request: RedditRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing Reddit search query")

    factory = RedditScraperFactory()
    scraper = factory.create_scraper()
    result = scraper.scrape(search_query)
    return {"message": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
