document.addEventListener('DOMContentLoaded', () => {
    const trendResultsDiv = document.getElementById('trendResults');
    const trendContentDiv = document.getElementById('trendContent');
    const statusMessagesDiv = document.getElementById('statusMessagesTrends');
    const pageTitleH1 = document.querySelector('.container h1'); // To update with search query

    const trendDataString = sessionStorage.getItem('trendAnalysisData');
    const searchQuery = sessionStorage.getItem('searchQueryForTrend');

    if (pageTitleH1 && searchQuery) {
        pageTitleH1.textContent += ` for "${searchQuery}"`;
    }


    if (trendDataString) {
        try {
            const trend_analysis_results = JSON.parse(trendDataString);
            let trendHTML = '';

            if(trend_analysis_results.detailed_predictions_csv_url) {
                trendHTML += `<p><a href="${trend_analysis_results.detailed_predictions_csv_url}" class="download-link" target="_blank" download>Download Detailed Trend Predictions CSV</a></p>`;
            }
            if(trend_analysis_results.cluster_summary_csv_url) {
                trendHTML += `<p><a href="${trend_analysis_results.cluster_summary_csv_url}" class="download-link" target="_blank" download>Download Trend Cluster Summary CSV</a></p>`;
            }

            if (trend_analysis_results.plot_urls && Object.keys(trend_analysis_results.plot_urls).length > 0) {
                trendHTML += '<h3>Trend Analysis Plots:</h3>';
                for (const [name_stem, url] of Object.entries(trend_analysis_results.plot_urls)) {
                     const displayName = formatDisplayNameFromTrendScript(name_stem, "Trend"); // Use a local or imported formatter
                     trendHTML += `<div class="plot-container"><p><strong>${displayName}</strong></p><img src="${url}?cb=${new Date().getTime()}" alt="${displayName}"></div>`;
                }
            } else if (trend_analysis_results.detailed_predictions_csv_url || trend_analysis_results.cluster_summary_csv_url) { 
                trendHTML += '<p class="info-message">No trend plots were generated for this analysis, but trend data (CSV) is available for download.</p>';
            }
            
            if (trendHTML) {
                trendResultsDiv.style.display = 'block';
                trendContentDiv.innerHTML = trendHTML;
            } else {
                statusMessagesDiv.innerHTML = '<div class="info-message">No trend analysis data or plots found for the previous analysis.</div>';
            }

            // Optional: Clear data from session storage after displaying if it's single-use
            // sessionStorage.removeItem('trendAnalysisData');
            // sessionStorage.removeItem('searchQueryForTrend');

        } catch (error) {
            statusMessagesDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> Could not load trend analysis data. It might be corrupted. ${error.message}</div>`;
            trendResultsDiv.style.display = 'none';
        }
    } else {
        statusMessagesDiv.innerHTML = '<div class="info-message">No analysis data found. Please <a href="index.html">start a new analysis</a>.</div>';
        trendResultsDiv.style.display = 'none';
    }
});

// Minimal formatDisplayName, assuming most complex logic is in script.js
// Or, you could share the function if using modules, or duplicate it.
function formatDisplayNameFromTrendScript(name_stem, typePrefix) {
    let displayName = name_stem
                        .replace(/^adv_|^ngram_|^plot_|^sentiment_|^trend_|^yt_/, "") 
                        .replace(/_/g, ' ') 
                        .replace('wordcloud', 'Word Cloud')
                        .replace('tfidf', 'TF-IDF')
                        .replace('rawfreq', 'Raw Frequency')
                        .replace('dist', 'Distribution')
                        .replace('vs', 'vs.')
                        .replace('elbow plot', 'Elbow Plot for Clustering')
                        .replace('cluster summary', 'Cluster Summary')
                        .replace('category distribution', 'Category Distribution');
    
    displayName = displayName.split(' ').map(word => {
        if (word.toLowerCase() === 'tfidf' || word.toLowerCase() === 'vs.') {
            return word.toUpperCase(); 
        }
        if (word) { 
            return word.charAt(0).toUpperCase() + word.slice(1);
        }
        return '';
    }).join(' ');
    return displayName.trim();
}