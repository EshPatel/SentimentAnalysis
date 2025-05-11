from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import pandas as pd
import os
import sys

# --- Path Setup for Imports ---
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT_DIR not in sys.path:
    sys.path.insert(0, APP_ROOT_DIR)
SENTIMENTAL_ANALYSIS_DIR = os.path.join(APP_ROOT_DIR, "sentimental_analysis")
if SENTIMENTAL_ANALYSIS_DIR not in sys.path:
     sys.path.insert(0, SENTIMENTAL_ANALYSIS_DIR)

# --- Module Imports ---
try:
    from factories.youtube_factory import YouTubeScraperFactory
except ImportError:
    print("WARNING: [app.py] factories.youtube_factory.YouTubeScraperFactory not found. Using Mock Scraper.")
    class MockYouTubeScraper:
        def scrape(self, search_query, video_limit=5):
            print(f"[app.py] Mock scraping for '{search_query}' with limit {video_limit}...")
            data = [{'Text': f"Comment {i+1} about {search_query}. It's great!"} for i in range(video_limit * 3)]
            data.append({'Text': f"A neutral comment on {search_query}."})
            data.append({'Text': f"This {search_query} video is bad."})
            if not data: return pd.DataFrame(columns=['Text'])
            return pd.DataFrame(data)
    class YouTubeScraperFactory:
        def create_scraper(self): return MockYouTubeScraper()

try:
    from sentimental_analysis.youtube_sentiment import perform_sentiment_analysis_and_generate_plots
except ImportError as e:
    print(f"ERROR: [app.py] Failed to import from youtube_sentiment: {e}")
    print(f"[app.py] Attempted to import from Python paths: {sys.path}")
    print("[app.py] Ensure sentimental_analysis/youtube_sentiment.py exists and is correctly structured.")
    def perform_sentiment_analysis_and_generate_plots(csv_input_path, output_plot_dir_param):
        print("WARNING: [app.py] Using DUMMY sentiment analysis function due to import error.")
        os.makedirs(output_plot_dir_param, exist_ok=True)
        dummy_plots = {}
        plot_stems_to_create = ["category_distribution", "positive_wordcloud_rawfreq", 
                                "negative_wordcloud_tfidf", "neutral_wordcloud_rawfreq"]
        for p_name_stem in plot_stems_to_create:
            p_filename = f"{p_name_stem}.png"
            p_path = os.path.join(output_plot_dir_param, p_filename)
            try:
                with open(p_path, "w") as f: f.write("dummy plot content")
                dummy_plots[p_name_stem] = p_path
            except Exception as file_e:
                print(f"[app.py] Could not create dummy plot file {p_path}: {file_e}")
        # Create a dummy analyzed CSV
        dummy_analyzed_csv_path = os.path.join(os.path.dirname(str(csv_input_path)), "analyzed_youtube_data.csv")
        try:
            pd.DataFrame({'Text':['dummy'], 'cleaned_text':['dummy'], 'sentiment_score':[0], 'sentiment_category':['Neutral']}).to_csv(dummy_analyzed_csv_path, index=False)
        except Exception as csv_e:
            print(f"[app.py] Could not create dummy analyzed CSV {dummy_analyzed_csv_path}: {csv_e}")
            dummy_analyzed_csv_path = None # Indicate failure
        return {"error": None, "plots": dummy_plots, "analyzed_csv_path": dummy_analyzed_csv_path}

app = FastAPI()

# --- Directory Definitions ---
BASE_DATA_DIR = os.path.join(APP_ROOT_DIR, "scrapers", "csv_outputs")
PRESENTATION_OUTPUT_DIR = os.path.join(APP_ROOT_DIR, "presentation_outputs")

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(PRESENTATION_OUTPUT_DIR, exist_ok=True)

SCRAPED_DATA_FILENAME = "youtube_data.csv"

class YouTubeRequest(BaseModel):
    search_query: str
    video_limit: Optional[int] = 1

@app.post("/analyze_youtube_sentiments/")
async def analyze_youtube_sentiments_endpoint(request: YouTubeRequest, background_tasks: BackgroundTasks):
    search_query = request.search_query
    video_limit = request.video_limit
    print(f"\n--- [app.py] Received request for query: '{search_query}', limit: {video_limit} ---")

    if not search_query:
        raise HTTPException(status_code=400, detail="Missing YouTube search query")

    raw_scraped_csv_path = os.path.join(BASE_DATA_DIR, SCRAPED_DATA_FILENAME)
    try:
        print(f"[app.py] Starting scraping for: '{search_query}'...")
        factory = YouTubeScraperFactory()
        scraper = factory.create_scraper()
        df_scraped_raw = scraper.scrape(search_query, video_limit=video_limit)
        
        print(f"[app.py] Columns from scraper output: {df_scraped_raw.columns.tolist()}")
        if 'cleaned_text' in df_scraped_raw.columns or 'sentiment_score' in df_scraped_raw.columns:
            print("[app.py] CRITICAL WARNING: Scraper is outputting already analyzed columns! This is incorrect.")
            print("[app.py] The scraper should ONLY output raw data (e.g., 'Text' column).")
            # Attempt to fix by dropping, but the scraper should be fixed.
            # df_scraped_raw = df_scraped_raw.drop(columns=['cleaned_text', 'sentiment_score', 'sentiment_category'], errors='ignore')

        if df_scraped_raw.empty:
            print(f"[app.py] Scraping resulted in an empty DataFrame for query: '{search_query}'.")
            raise HTTPException(status_code=404, detail="No comments found or error during scraping.")
        
        df_scraped_raw.to_csv(raw_scraped_csv_path, index=False)
        print(f"[app.py] Scraping complete. RAW data ({len(df_scraped_raw)} rows) saved to: {raw_scraped_csv_path}")
    except Exception as e:
        print(f"[app.py] Scraping error: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

    print(f"[app.py] Starting sentiment analysis. Input: {raw_scraped_csv_path}, Plot Output Dir: {PRESENTATION_OUTPUT_DIR}")
    analysis_results = perform_sentiment_analysis_and_generate_plots(raw_scraped_csv_path, PRESENTATION_OUTPUT_DIR)

    if analysis_results.get("error"):
        print(f"[app.py] Sentiment analysis failed: {analysis_results['error']}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {analysis_results['error']}")

    print("[app.py] Sentiment analysis process complete.")
    plot_paths_dict = analysis_results.get("plots", {}) # e.g., {'category_distribution': 'path/to/plot.png', 'positive_wordcloud_rawfreq': '...'}
    analyzed_csv_full_path = analysis_results.get("analyzed_csv_path")

    response_data = {
        "message": "Analysis complete.",
        "scraped_data_url": f"/download/{SCRAPED_DATA_FILENAME}",
        "analyzed_data_url": f"/download/{os.path.basename(str(analyzed_csv_full_path))}" if analyzed_csv_full_path else None,
        "plot_urls": {
            name_stem: f"/plots/{os.path.basename(str(full_path))}" for name_stem, full_path in plot_paths_dict.items() if full_path # Ensure path is not None
        }
    }
    print(f"[app.py] Sending response: {response_data}")
    return JSONResponse(content=response_data)

@app.get("/download/{filename}")
async def download_file(filename: str):
    target_dir = os.path.abspath(BASE_DATA_DIR)
    file_path = os.path.abspath(os.path.join(target_dir, filename))
    print(f"[app.py] Download request for: {file_path} (Allowed dir: {target_dir})")
    if os.path.commonprefix((file_path, target_dir)) == target_dir and os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/csv', filename=filename)
    print(f"[app.py] Download FAILED for: {filename}. Path: {file_path}. Exists: {os.path.exists(file_path)}")
    raise HTTPException(status_code=404, detail=f"File '{filename}' not found or access denied.")

@app.get("/plots/{filename}")
async def get_plot_image(filename: str):
    target_dir = os.path.abspath(PRESENTATION_OUTPUT_DIR)
    file_path = os.path.abspath(os.path.join(target_dir, filename))
    print(f"[app.py] Plot request for: {file_path} (Allowed dir: {target_dir})")
    if os.path.commonprefix((file_path, target_dir)) == target_dir and os.path.exists(file_path):
        return FileResponse(file_path, media_type='image/png')
    print(f"[app.py] Plot serving FAILED for: {filename}. Path: {file_path}. Exists: {os.path.exists(file_path)}")
    raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found or access denied.")

@app.get("/", response_class=HTMLResponse)
async def main_page_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Sentiment Analyzer</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f0f2f5; color: #1c1e21; line-height: 1.6; }
            .container { background-color: #ffffff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 8px 16px rgba(0,0,0,0.1); max-width: 800px; margin: 40px auto; }
            h1 { color: #1877f2; text-align: center; margin-bottom: 20px; }
            h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-top: 30px; }
            h3 { color: #555; margin-top: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #606770; }
            input[type="text"], input[type="number"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ccd0d5; border-radius: 6px; font-size: 16px; }
            button { background-color: #1877f2; color: white; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background-color 0.2s; }
            button:hover { background-color: #166fe5; }
            #results { margin-top: 25px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #f7f8fa; }
            #results p, #results a { margin: 8px 0; font-size: 15px; }
            #results a { color: #1877f2; text-decoration: none; }
            #results a:hover { text-decoration: underline; }
            #results img { max-width: 100%; height: auto; border: 1px solid #dddfe2; margin-top: 10px; margin-bottom: 20px; border-radius: 6px; display: block; }
            .loader { border: 5px solid #f0f2f5; border-top: 5px solid #1877f2; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .error-message { color: #fa383e; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YouTube Sentiment Analyzer</h1>
            <form id="analyzeForm">
                <label for="search_query">Search Query:</label>
                <input type="text" id="search_query" name="search_query" value="positive thinking" required>
                <label for="video_limit">Video Limit (for scraping):</label>
                <input type="number" id="video_limit" name="video_limit" value="1" min="1">
                <button type="button" onclick="submitAnalysis()">Analyze Sentiments</button>
            </form>
            <div id="results"><p>Enter a query and click "Analyze Sentiments" to see the results.</p></div>
        </div>
        <script>
            async function submitAnalysis() {
                const form = document.getElementById('analyzeForm');
                const resultsDiv = document.getElementById('results');
                const data = {
                    search_query: form.search_query.value,
                    video_limit: parseInt(form.video_limit.value)
                };
                resultsDiv.innerHTML = '<div class="loader"></div><p style="text-align:center;">Processing your request... This may take a few moments depending on the video limit.</p>';
                try {
                    const response = await fetch('/analyze_youtube_sentiments/', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await response.json(); 
                    if (response.ok) {
                        let html = '<h2>Analysis Results:</h2>';
                        html += `<p>${result.message}</p>`;
                        if(result.scraped_data_url) html += `<p><a href="${result.scraped_data_url}" target="_blank" download>Download Scraped CSV</a></p>`;
                        if(result.analyzed_data_url) html += `<p><a href="${result.analyzed_data_url}" target="_blank" download>Download Analyzed CSV</a></p>`;
                        
                        if (result.plot_urls && Object.keys(result.plot_urls).length > 0) {
                            html += '<h3>Generated Plots:</h3>';
                            for (const [name_stem, url] of Object.entries(result.plot_urls)) {
                                const displayName = name_stem.replace(/_/g, ' ').replace('wordcloud', 'Word Cloud').replace('tfidf', '(TF-IDF)').replace('rawfreq', '(Raw Freq)').replace(/(?:^|\\s)\\S/g, a => a.toUpperCase());
                                html += `<p><strong>${displayName.charAt(0).toUpperCase() + displayName.slice(1)}:</strong></p><img src="${url}?cache_bust=${new Date().getTime()}" alt="${displayName}"><br>`;
                            }
                        } else {
                            html += '<p>No plots were generated or returned.</p>';
                        }
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = `<p class="error-message"><strong>Error ${response.status}:</strong> ${result.detail || 'An unknown error occurred processing your request.'}</p>`;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<p class="error-message"><strong>Network or Client-Side Error:</strong> ${error}. Please check the browser console for more details.</p>`;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    print(f"--- [app.py] Starting Application ---")
    print(f"[app.py] Application Root Directory: {APP_ROOT_DIR}")
    print(f"[app.py] Data Directory for CSVs (BASE_DATA_DIR): {BASE_DATA_DIR}")
    print(f"[app.py] Directory for Plot Outputs (PRESENTATION_OUTPUT_DIR): {PRESENTATION_OUTPUT_DIR}")
    # print(f"[app.py] Current Python Path: {sys.path}") # Can be very verbose
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)