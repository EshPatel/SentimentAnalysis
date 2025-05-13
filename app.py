import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import pandas as pd
import os
import sys
import shutil
from fastapi.staticfiles import StaticFiles

APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT_DIR not in sys.path:
    sys.path.insert(0, APP_ROOT_DIR)


try:
    from factories.youtube_factory import YouTubeScraperFactory
except ImportError as e_scraper:
    print(f"WARNING: [app.py] factories.youtube_factory.YouTubeScraperFactory not found: {e_scraper}. Using Mock Scraper.")
    class MockYouTubeScraper: 
        def scrape(self, search_query, video_limit=5):
            print(f"[app.py] Mock scraping for '{search_query}' with limit {video_limit}...")
            from datetime import datetime, timedelta
            base_date = datetime.now()
            data = []
            for i in range(video_limit * 5):
                data.append({
                    'Text': f"Mock comment {i+1} on {search_query}. This topic is great! {'AMAZING!!!' if i%3==0 else ''}",
                    'PublishedAt': (base_date - timedelta(days=i % 10)).isoformat()
                })
            return pd.DataFrame(data)
    class YouTubeScraperFactory:
        def create_scraper(self): return MockYouTubeScraper()

# Sentiment Analysis Module (assuming youtube_sentiment.py is in the same dir as app.py)
try:
    from sentimental_analysis.youtube_sentiment import perform_sentiment_analysis_and_generate_plots
    SENTIMENT_MODULE_LOADED = True
    print("[app.py] Successfully imported 'perform_sentiment_analysis_and_generate_plots' from 'youtube_sentiment.py'")
except ImportError as e_sentiment:
    SENTIMENT_MODULE_LOADED = False
    print(f"ERROR: [app.py] Failed to import from 'youtube_sentiment.py': {e_sentiment}")
    print(f"[app.py] Ensure 'youtube_sentiment.py' is in the root directory with 'app.py'. Current sys.path: {sys.path}")
    def perform_sentiment_analysis_and_generate_plots(csv, plot_dir): # Dummy
        print("WARNING: [app.py] Using DUMMY sentiment analysis function.")
        analyzed_csv_path = os.path.join(os.path.dirname(str(csv)), "analyzed_youtube_data.csv") # Expected by the trend_analysis.py you provided
        return {"error": None, "plots": {"category_distribution": "dummy_cat.png"}, "analyzed_csv_path": analyzed_csv_path}

# Trend Analysis Module (assuming trend_analysis.py is in the same dir as app.py)
try:
    from sentimental_analysis.trend_analysis import perform_trend_analysis # Ensure this function exists in your trend_analysis.py
    TREND_ANALYSIS_MODULE_LOADED = True
    print("[app.py] Successfully imported 'perform_trend_analysis' from 'trend_analysis.py'")
except ImportError as e_trend:
    TREND_ANALYSIS_MODULE_LOADED = False
    print(f"ERROR: [app.py] Failed to import 'perform_trend_analysis' from 'trend_analysis.py': {e_trend}")
    print(f"[app.py] Ensure 'trend_analysis.py' is in the root directory and has the function. Current sys.path: {sys.path}")
    def perform_trend_analysis(analyzed_sentiment_csv_path, output_plot_dir_param, timestamp_col=None): # Dummy
        print("WARNING: [app.py] Using DUMMY trend analysis function.")
        dummy_trend_plots = {"elbow_plot": "dummy_elbow.png"} # Keep it simple
        detailed_csv = os.path.join(os.path.dirname(str(analyzed_sentiment_csv_path)), "advanced_trend_analysis_predictions.csv")
        summary_csv = os.path.join(output_plot_dir_param, "adv_cluster_trend_summary.csv")
        return {"error": None, "plots": dummy_trend_plots, "detailed_predictions_csv_path": detailed_csv, "cluster_summary_csv_path": summary_csv}

app = FastAPI()

# --- Directory Definitions ---
BASE_DATA_DIR = os.path.join(APP_ROOT_DIR, "scrapers", "csv_outputs")
PRESENTATION_OUTPUT_DIR = os.path.join(APP_ROOT_DIR, "presentation_outputs_app")

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(PRESENTATION_OUTPUT_DIR, exist_ok=True)

RAW_SCRAPED_DATA_FILENAME = "youtube_data_raw.csv"
# This filename is produced by the youtube_sentiment.py you provided
SENTIMENT_ANALYZED_FILENAME = "analyzed_youtube_data.csv"
# These filenames are produced by the trend_analysis.py you provided
TREND_ANALYSIS_DETAILED_CSV_OUTPUT_FILENAME = 'advanced_trend_analysis_predictions.csv'
TREND_ANALYSIS_SUMMARY_CSV_FILENAME = 'adv_cluster_trend_summary.csv'

TIMESTAMP_COLUMN_FOR_TRENDS = 'PublishedAt'

class YouTubeRequest(BaseModel):
    search_query: str
    video_limit: Optional[int] = 1

@app.post("/analyze_youtube_full_pipeline/")
async def analyze_youtube_full_pipeline_endpoint(request: YouTubeRequest, background_tasks: BackgroundTasks):
    search_query = request.search_query
    video_limit = request.video_limit
    print(f"\n--- [app.py] FULL PIPELINE Request: Query='{search_query}', Limit={video_limit} ---")

    if not search_query:
        raise HTTPException(status_code=400, detail="Missing YouTube search query")

    raw_scraped_csv_path = os.path.join(BASE_DATA_DIR, RAW_SCRAPED_DATA_FILENAME)
    try:
        print(f"[app.py] Starting scraping...")
        factory = YouTubeScraperFactory()
        scraper = factory.create_scraper()
        df_scraped_raw = scraper.scrape(search_query, video_limit=video_limit)
        
        if df_scraped_raw.empty:
            raise HTTPException(status_code=404, detail="No comments found or error during scraping.")
        df_scraped_raw.to_csv(raw_scraped_csv_path, index=False)
        print(f"[app.py] Scraping complete. RAW data saved to: {raw_scraped_csv_path}")
    except Exception as e:
        print(f"[app.py] Scraping error: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

    sentiment_analysis_results = {}
    analyzed_sentiment_csv_path = None
    if SENTIMENT_MODULE_LOADED:
        print(f"[app.py] Starting sentiment analysis...")
        sentiment_analysis_results = perform_sentiment_analysis_and_generate_plots(
            raw_scraped_csv_path, PRESENTATION_OUTPUT_DIR
        )
        if sentiment_analysis_results.get("error"):
            raise HTTPException(status_code=500, detail=f"Sentiment analysis part failed: {sentiment_analysis_results['error']}")
        analyzed_sentiment_csv_path = sentiment_analysis_results.get("analyzed_csv_path")
        print(f"[app.py] Sentiment analysis complete. Analyzed CSV: {analyzed_sentiment_csv_path}")
    else:
        return JSONResponse(status_code=501, content={"detail": "Sentiment analysis module 'youtube_sentiment.py' not loaded."})

    trend_analysis_results = {}
    detailed_trend_predictions_csv_path = None
    cluster_trend_summary_csv_path = None
    if TREND_ANALYSIS_MODULE_LOADED and analyzed_sentiment_csv_path and os.path.exists(analyzed_sentiment_csv_path):
        print(f"[app.py] Starting trend analysis using: {analyzed_sentiment_csv_path}...")
        trend_analysis_results = perform_trend_analysis( # Call the refactored function
            analyzed_sentiment_csv_path, PRESENTATION_OUTPUT_DIR, timestamp_col_name=TIMESTAMP_COLUMN_FOR_TRENDS
        )
        if trend_analysis_results.get("error"):
            print(f"[app.py] Trend analysis failed: {trend_analysis_results['error']}")
        else:
            detailed_trend_predictions_csv_path = trend_analysis_results.get("detailed_predictions_csv_path")
            cluster_trend_summary_csv_path = trend_analysis_results.get("cluster_summary_csv_path")
            print(f"[app.py] Trend analysis complete.")
    elif not TREND_ANALYSIS_MODULE_LOADED:
        print("[app.py] Trend analysis module 'trend_analysis.py' not loaded. Skipping.")
    elif not analyzed_sentiment_csv_path or not os.path.exists(analyzed_sentiment_csv_path):
        print("[app.py] Analyzed sentiment CSV not available for trend analysis. Skipping.")

    response_data = {
        "message": "Full analysis pipeline executed.",
        "scraped_data_url": f"/download/raw_data/{RAW_SCRAPED_DATA_FILENAME}",
        "sentiment_results": {
            "analyzed_data_url": f"/download/sentiment_analyzed/{SENTIMENT_ANALYZED_FILENAME}" if analyzed_sentiment_csv_path else None,
            "plot_urls": {
                name_stem: f"/plots/sentiment/{os.path.basename(str(full_path))}" 
                for name_stem, full_path in sentiment_analysis_results.get("plots", {}).items() if full_path
            }
        },
        "trend_analysis_results": {
            "detailed_predictions_csv_url": f"/download/trend_predictions/{TREND_ANALYSIS_DETAILED_CSV_OUTPUT_FILENAME}" if detailed_trend_predictions_csv_path else None,
            "cluster_summary_csv_url": f"/download/trend_summary/{TREND_ANALYSIS_SUMMARY_CSV_FILENAME}" if cluster_trend_summary_csv_path else None,
            "plot_urls": {
                name_stem: f"/plots/trends/{os.path.basename(str(full_path))}" 
                for name_stem, full_path in trend_analysis_results.get("plots", {}).items() if full_path
            }
        }
    }
    print(f"[app.py] Sending response for full pipeline.")
    return JSONResponse(content=response_data)

async def serve_file(filename: str, directory: str, media_type: str):
    target_dir = os.path.abspath(directory)
    file_path = os.path.abspath(os.path.join(target_dir, filename))
    print(f"[app.py] File request: {file_path} (Allowed dir: {target_dir})")
    if os.path.commonprefix((file_path, target_dir)) == target_dir and os.path.exists(file_path):
        return FileResponse(file_path, media_type=media_type, filename=filename)
    print(f"[app.py] File serving FAILED for: {filename}. Path: {file_path}. Exists: {os.path.exists(file_path)}")
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found or access denied.")

@app.get("/download/raw_data/{filename}")
async def download_raw_data_file(filename: str):
    return await serve_file(filename, BASE_DATA_DIR, 'text/csv')

@app.get("/download/sentiment_analyzed/{filename}")
async def download_sentiment_analyzed_file(filename: str):
    return await serve_file(filename, BASE_DATA_DIR, 'text/csv')

@app.get("/download/trend_predictions/{filename}")
async def download_trend_prediction_file(filename: str):
    return await serve_file(filename, BASE_DATA_DIR, 'text/csv')

@app.get("/download/trend_summary/{filename}")
async def download_trend_summary_file(filename: str):
    return await serve_file(filename, PRESENTATION_OUTPUT_DIR, 'text/csv')

@app.get("/plots/sentiment/{filename}")
async def get_sentiment_plot_image(filename: str):
    return await serve_file(filename, PRESENTATION_OUTPUT_DIR, 'image/png')

@app.get("/plots/trends/{filename}")
async def get_trend_plot_image(filename: str):
    return await serve_file(filename, PRESENTATION_OUTPUT_DIR, 'image/png')

# @app.get("/", response_class=HTMLResponse)
# async def main_page_html():
#     # HTML is the same as the previous one, designed for the nested response
#     return """
    
#     """
app.mount("/", StaticFiles(directory="static",html = True), name="static")

if __name__ == "__main__":
    print(f"--- [app.py] Starting Application ---")
    print(f"[app.py] Current Working Directory: {os.getcwd()}")
    print(f"[app.py] APP_ROOT_DIR (Directory of app.py): {APP_ROOT_DIR}")
    print(f"[app.py] Python sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    
    print(f"[app.py] Checking for 'youtube_sentiment.py' in APP_ROOT_DIR: {os.path.exists(os.path.join(APP_ROOT_DIR, 'youtube_sentiment.py'))}")
    print(f"[app.py] Checking for 'trend_analysis.py' in APP_ROOT_DIR: {os.path.exists(os.path.join(APP_ROOT_DIR, 'trend_analysis.py'))}")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)