from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from factories.youtube_factory import YouTubeScraperFactory

app = FastAPI()

# Request body models
class YouTubeRequest(BaseModel):
    search_query: str

# Routes
@app.post('/scrape/youtube')
async def scrape_youtube(request: YouTubeRequest):
    search_query = request.search_query
    if not search_query:
        raise HTTPException(status_code=400, detail="Missing YouTube search query")

    factory = YouTubeScraperFactory()
    scraper = factory.create_scraper()
    result = scraper.scrape(search_query)
    
    # Ensure the result is JSON serializable (e.g., convert to dict if necessary)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
