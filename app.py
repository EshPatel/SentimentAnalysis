from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
import pandas as pd
import os
import sys
import shutil

# --- Path Setup for Imports ---
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT_DIR not in sys.path:
    sys.path.insert(0, APP_ROOT_DIR)

# --- Module Imports ---
# Scraper Factory
try:
    from factories.youtube_factory import YouTubeScraperFactory
except ImportError:
    print("WARNING: [app.py] factories.youtube_factory.YouTubeScraperFactory not found. Using Mock Scraper.")
    class MockYouTubeScraper:
        def scrape(self, search_query, video_limit=5):
            print(f"[app.py] Mock scraping for '{search_query}' with limit {video_limit}...")
            from datetime import datetime, timedelta
            base_date = datetime.now()
            data = []
            for i in range(video_limit * 5):
                data.append({
                    'Text': f"Comment {i+1} on {search_query}. This is a comment about topic_A and keyword1. {'GREAT!!!' if i%3==0 else ''}",
                    'PublishedAt': (base_date - timedelta(days=i % 10)).isoformat() # Mock timestamps
                })
            data.append({'Text': f"Another neutral thought on {search_query}, discussing topic_B and keyword2.", 'PublishedAt': (base_date - timedelta(days=3)).isoformat()})
            data.append({'Text': f"This {search_query} is not good. Topic_C and keyword3. {'BAD!!!' if True else ''}", 'PublishedAt': (base_date - timedelta(days=1)).isoformat()})
            if not data: return pd.DataFrame(columns=['Text', 'PublishedAt'])
            return pd.DataFrame(data)
    class YouTubeScraperFactory:
        def create_scraper(self): return MockYouTubeScraper()

# Sentiment Analysis Module (using the version from THIS prompt)
try:
    from youtube_sentiment import perform_sentiment_analysis_and_generate_plots
    SENTIMENT_MODULE_LOADED = True
except ImportError as e:
    SENTIMENT_MODULE_LOADED = False
    print(f"ERROR: [app.py] Failed to import from youtube_sentiment: {e}")
    def perform_sentiment_analysis_and_generate_plots(csv, plot_dir): # Dummy
        print("WARNING: [app.py] Using DUMMY sentiment analysis function.")
        # This dummy needs to create 'analyzed_youtube_data.csv' and use 'cleaned_text'
        analyzed_csv_path = os.path.join(os.path.dirname(str(csv)), "analyzed_youtube_data.csv")
        try:
            pd.DataFrame({'Text':['dummy'], 'cleaned_text':['dummy text'], 'sentiment_score':[0], 'sentiment_category':['Neutral']}).to_csv(analyzed_csv_path, index=False)
        except: pass
        return {"error": None, "plots": {"category_distribution": "dummy_cat.png", "positive_wordcloud_rawfreq":"dummy_pos.png"}, "analyzed_csv_path": analyzed_csv_path}

# Trend Analysis Module (assuming you refactor the one from THIS prompt)
# Let's name the refactored function `perform_basic_trend_analysis` to distinguish
try:
    # YOU NEED TO CREATE THIS FUNCTION IN YOUR trend_analysis.py
    from trend_analysis import perform_trend_analysis 
    TREND_ANALYSIS_MODULE_LOADED = True
except ImportError as e:
    TREND_ANALYSIS_MODULE_LOADED = False
    print(f"ERROR: [app.py] Failed to import perform_trend_analysis from trend_analysis.py: {e}")
    print("[app.py] Ensure trend_analysis.py has a callable function 'perform_trend_analysis'.")
    def perform_trend_analysis(analyzed_sentiment_csv_path, output_plot_dir_param, timestamp_col=None): # Dummy
        print("WARNING: [app.py] Using DUMMY trend analysis function.")
        dummy_trend_plots = {}
        trend_plot_stems = ["adv_elbow_plot", "adv_cluster_trend_score_comparison", "adv_network_cluster_0"] # Match filenames from that trend.py
        for stem in trend_plot_stems:
            p_filename = f"{stem}.png"
            p_path = os.path.join(output_plot_dir_param, p_filename)
            try:
                with open(p_path, "w") as f: f.write("dummy trend plot")
                dummy_trend_plots[stem.replace("adv_", "")] = p_path # Simplify key for HTML
            except Exception as file_e: print(f"[app.py] Could not create dummy trend plot file {p_path}: {file_e}")
        
        dummy_trend_detailed_csv_path = os.path.join(os.path.dirname(str(analyzed_sentiment_csv_path)), "advanced_trend_analysis_predictions.csv")
        dummy_trend_summary_csv_path = os.path.join(output_plot_dir_param, "adv_cluster_trend_summary.csv")
        try:
            pd.DataFrame({'Cluster':[0], 'TrendScore_Normalized':[75]}).to_csv(dummy_trend_detailed_csv_path, index=False)
            pd.DataFrame({'Cluster':[0], 'TopTerms':['dummy trend']}).to_csv(dummy_trend_summary_csv_path, index=False)
        except: pass
        return {
            "error": None, 
            "plots": dummy_trend_plots, 
            "detailed_predictions_csv_path": dummy_trend_detailed_csv_path,
            "cluster_summary_csv_path": dummy_trend_summary_csv_path
        }

app = FastAPI()

BASE_DATA_DIR = os.path.join(APP_ROOT_DIR, "scrapers", "csv_outputs")
PRESENTATION_OUTPUT_DIR = os.path.join(APP_ROOT_DIR, "presentation_outputs_app") # Main plot dir

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(PRESENTATION_OUTPUT_DIR, exist_ok=True)

RAW_SCRAPED_DATA_FILENAME = "youtube_data_raw.csv"
# Filename from the youtube_sentiment.py you provided in THIS prompt
SENTIMENT_ANALYZED_FILENAME = "analyzed_youtube_data.csv"

# Filenames from the trend_analysis.py you provided in THIS prompt
TREND_ANALYSIS_DETAILED_CSV_OUTPUT_FILENAME = 'advanced_trend_analysis_predictions.csv' # Saved in BASE_DATA_DIR
TREND_ANALYSIS_SUMMARY_CSV_FILENAME = 'adv_cluster_trend_summary.csv' # Saved in PRESENTATION_OUTPUT_DIR

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
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

    sentiment_analysis_results = {}
    analyzed_sentiment_csv_path = None # This will be BASE_DATA_DIR/analyzed_youtube_data.csv
    if SENTIMENT_MODULE_LOADED:
        print(f"[app.py] Starting sentiment analysis...")
        sentiment_analysis_results = perform_sentiment_analysis_and_generate_plots(
            raw_scraped_csv_path, PRESENTATION_OUTPUT_DIR # Plots go to presentation_outputs_app
        )
        if sentiment_analysis_results.get("error"):
            raise HTTPException(status_code=500, detail=f"Sentiment analysis part failed: {sentiment_analysis_results['error']}")
        analyzed_sentiment_csv_path = sentiment_analysis_results.get("analyzed_csv_path") # This is crucial
        print(f"[app.py] Sentiment analysis complete. Analyzed CSV: {analyzed_sentiment_csv_path}")
    else:
        return JSONResponse(status_code=501, content={"detail": "Sentiment analysis module not loaded."})

    trend_analysis_results = {}
    detailed_trend_predictions_csv_path = None
    cluster_trend_summary_csv_path = None
    if TREND_ANALYSIS_MODULE_LOADED and analyzed_sentiment_csv_path and os.path.exists(analyzed_sentiment_csv_path):
        print(f"[app.py] Starting trend analysis using: {analyzed_sentiment_csv_path}...")
        # The trend_analysis.py you provided saves plots to its own "presentation_outputs_advanced_trends"
        # For consistency, we'll pass PRESENTATION_OUTPUT_DIR to it as well.
        # You'll need to ensure your refactored trend_analysis.py uses this parameter.
        trend_analysis_results = perform_trend_analysis( # Call the refactored function
            analyzed_sentiment_csv_path, PRESENTATION_OUTPUT_DIR, timestamp_col=TIMESTAMP_COLUMN_FOR_TRENDS
        )
        if trend_analysis_results.get("error"):
            print(f"[app.py] Trend analysis failed: {trend_analysis_results['error']}")
        else:
            # Assuming your refactored trend_analysis returns these paths
            detailed_trend_predictions_csv_path = trend_analysis_results.get("detailed_predictions_csv_path") # Expected in BASE_DATA_DIR
            cluster_trend_summary_csv_path = trend_analysis_results.get("cluster_summary_csv_path")       # Expected in PRESENTATION_OUTPUT_DIR
            print(f"[app.py] Trend analysis complete.")
    elif not TREND_ANALYSIS_MODULE_LOADED:
        print("[app.py] Trend analysis module not loaded. Skipping.")
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

@app.get("/", response_class=HTMLResponse)
async def main_page_html():
    # HTML needs to be updated to show both sentiment and trend results correctly
    # (The HTML from the previous response is already good for this nested structure)
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>YouTube Advanced Analyzer</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f0f2f5; color: #1c1e21; line-height: 1.6; }
            .container { background-color: #ffffff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 8px 16px rgba(0,0,0,0.1); max-width: 900px; margin: 40px auto; }
            h1 { color: #1877f2; text-align: center; margin-bottom: 20px; }
            h2 { color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-top: 30px; font-size: 1.5em; }
            h3 { color: #555; margin-top: 20px; font-size: 1.2em; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #606770; }
            input[type="text"], input[type="number"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ccd0d5; border-radius: 6px; font-size: 16px; }
            button { background-color: #1877f2; color: white; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background-color 0.2s; }
            button:hover { background-color: #166fe5; }
            .results-section { margin-top: 25px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #f7f8fa; }
            .results-section p, .results-section a { margin: 8px 0; font-size: 15px; }
            .results-section a { color: #1877f2; text-decoration: none; font-weight: bold; }
            .results-section a:hover { text-decoration: underline; }
            .results-section img { max-width: 100%; height: auto; border: 1px solid #dddfe2; margin-top: 10px; margin-bottom: 20px; border-radius: 6px; display: block; }
            .plot-container { margin-bottom: 20px; }
            .loader { border: 5px solid #f0f2f5; border-top: 5px solid #1877f2; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .error-message { color: #fa383e; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YouTube Advanced Analyzer (Sentiment & Trends)</h1>
            <form id="analyzeForm">
                <label for="search_query">Search Query:</label>
                <input type="text" id="search_query" name="search_query" value="AI in education" required>
                <label for="video_limit">Video Limit (for scraping):</label>
                <input type="number" id="video_limit" name="video_limit" value="1" min="1">
                <button type="button" onclick="submitFullAnalysis()">Run Full Analysis</button>
            </form>
            <div id="processingMessage"></div>
            <div id="sentimentResults" class="results-section" style="display:none;"><h2>Sentiment Analysis Results</h2><div id="sentimentContent"></div></div>
            <div id="trendResults" class="results-section" style="display:none;"><h2>Trend Analysis Results</h2><div id="trendContent"></div></div>
        </div>
        <script>
            async function submitFullAnalysis() {
                const form = document.getElementById('analyzeForm');
                const processingDiv = document.getElementById('processingMessage');
                const sentimentResultsDiv = document.getElementById('sentimentResults');
                const trendResultsDiv = document.getElementById('trendResults');
                const sentimentContentDiv = document.getElementById('sentimentContent');
                const trendContentDiv = document.getElementById('trendContent');

                const data = {
                    search_query: form.search_query.value,
                    video_limit: parseInt(form.video_limit.value)
                };

                processingDiv.innerHTML = '<div class="loader"></div><p style="text-align:center;">Processing your request... This may take several moments.</p>';
                sentimentResultsDiv.style.display = 'none';
                trendResultsDiv.style.display = 'none';
                sentimentContentDiv.innerHTML = '';
                trendContentDiv.innerHTML = '';

                try {
                    const response = await fetch('/analyze_youtube_full_pipeline/', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await response.json(); 
                    processingDiv.innerHTML = ''; 

                    if (response.ok) {
                        let sentimentHTML = '';
                        if (result.sentiment_results) {
                            sentimentResultsDiv.style.display = 'block';
                            if(result.sentiment_results.analyzed_data_url) sentimentHTML += `<p><a href="${result.sentiment_results.analyzed_data_url}" target="_blank" download>Download Sentiment Analyzed CSV</a></p>`;
                            if (result.sentiment_results.plot_urls && Object.keys(result.sentiment_results.plot_urls).length > 0) {
                                sentimentHTML += '<h3>Sentiment Plots:</h3>';
                                for (const [name_stem, url] of Object.entries(result.sentiment_results.plot_urls)) {
                                    const displayName = formatDisplayName(name_stem, "Sentiment");
                                    sentimentHTML += `<div class="plot-container"><p><strong>${displayName}:</strong></p><img src="${url}?cb=${new Date().getTime()}" alt="${displayName}"></div>`;
                                }
                            } else { sentimentHTML += '<p>No sentiment plots generated.</p>'; }
                            sentimentContentDiv.innerHTML = sentimentHTML;
                        }

                        let trendHTML = '';
                        if (result.trend_analysis_results) {
                            trendResultsDiv.style.display = 'block';
                            if(result.trend_analysis_results.detailed_predictions_csv_url) trendHTML += `<p><a href="${result.trend_analysis_results.detailed_predictions_csv_url}" target="_blank" download>Download Detailed Trend Predictions CSV</a></p>`;
                            if(result.trend_analysis_results.cluster_summary_csv_url) trendHTML += `<p><a href="${result.trend_analysis_results.cluster_summary_csv_url}" target="_blank" download>Download Trend Cluster Summary CSV</a></p>`;
                            if (result.trend_analysis_results.plot_urls && Object.keys(result.trend_analysis_results.plot_urls).length > 0) {
                                trendHTML += '<h3>Trend Analysis Plots:</h3>';
                                for (const [name_stem, url] of Object.entries(result.trend_analysis_results.plot_urls)) {
                                     const displayName = formatDisplayName(name_stem, "Trend");
                                     trendHTML += `<div class="plot-container"><p><strong>${displayName}:</strong></p><img src="${url}?cb=${new Date().getTime()}" alt="${displayName}"></div>`;
                                }
                            } else { trendHTML += '<p>No trend plots generated.</p>'; }
                            trendContentDiv.innerHTML = trendHTML;
                        }
                        if (!sentimentHTML && !trendHTML && !result.message.includes("Error")) { // Check if no useful content and no explicit error message
                             processingDiv.innerHTML = "<p>Analysis may have completed, but no specific results or plots were returned. Check server logs.</p>"
                        } else if (!sentimentHTML && !trendHTML && result.message) { // Display message if it's the only thing
                            processingDiv.innerHTML = `<p>${result.message}</p>`;
                        }


                    } else {
                        processingDiv.innerHTML = '';
                        sentimentResultsDiv.style.display = 'block';
                        sentimentContentDiv.innerHTML = `<p class="error-message"><strong>Error ${response.status}:</strong> ${result.detail || 'An unknown error occurred.'}</p>`;
                    }
                } catch (error) {
                    processingDiv.innerHTML = '';
                    sentimentResultsDiv.style.display = 'block';
                    sentimentContentDiv.innerHTML = `<p class="error-message"><strong>Network/Client Error:</strong> ${error}. Check console.</p>`;
                }
            }

            function formatDisplayName(name_stem, typePrefix) {
                let displayName = name_stem
                                    .replace("adv_", "")
                                    .replace("ngram_", "")
                                    .replace(/_/g, ' ')
                                    .replace('wordcloud', 'Word Cloud')
                                    .replace('tfidf', '(TF-IDF)')
                                    .replace('rawfreq', '(Raw Freq)')
                                    .replace('cluster ', 'Cluster ')
                                    .replace('category distribution', 'Category Distribution');
                
                // Capitalize first letter of each word
                displayName = displayName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
                return displayName;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    print(f"--- [app.py] Starting Application ---")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)