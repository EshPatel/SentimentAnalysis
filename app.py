from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import io
import uvicorn
import pandas as pd
from pydantic import BaseModel
from factories.youtube_factory import YouTubeScraperFactory

app = FastAPI()

# Request body models
class YouTubeRequest(BaseModel):
    search_query: str
    video_limit: Optional[int] = None

# Routes
@app.post('/scrape/youtube')
async def scrape_youtube(request: YouTubeRequest):
    search_query = request.search_query
    video_limit = request.video_limit

    if not search_query:
        raise HTTPException(status_code=400, detail="Missing YouTube search query")

    factory = YouTubeScraperFactory()
    scraper = factory.create_scraper()
    df = scraper.scrape(search_query, video_limit)

    # Convert DataFrame to CSV in memory
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    # Return the CSV as a downloadable file
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=youtube_data.csv"}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
