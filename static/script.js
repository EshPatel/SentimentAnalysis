async function submitFullAnalysis() {
    const form = document.getElementById('analyzeForm');
    const analyzeButton = document.getElementById('analyzeButton');
    const statusMessagesDiv = document.getElementById('statusMessages');
    const sentimentResultsDiv = document.getElementById('sentimentResults');
    const sentimentContentDiv = document.getElementById('sentimentContent');
    const ctaTrendAnalysisContainer = document.getElementById('ctaTrendAnalysisContainer');
    const goToTrendAnalysisButton = document.getElementById('goToTrendAnalysisButton');

    if (!form.search_query.value.trim()) {
        statusMessagesDiv.innerHTML = `<div class="error-message"><strong>Input Error:</strong> Please enter a YouTube Search Query.</div>`;
        form.search_query.focus();
        return;
    }
    const videoLimit = parseInt(form.video_limit.value);
    if (isNaN(videoLimit) || videoLimit < 1) {
        statusMessagesDiv.innerHTML = `<div class="error-message"><strong>Input Error:</strong> Number of videos must be a valid number and at least 1.</div>`;
        form.video_limit.focus();
        return;
    }

    const data = {
        search_query: form.search_query.value,
        video_limit: videoLimit
    };

    analyzeButton.disabled = true;
    analyzeButton.setAttribute('aria-busy', 'true');
    statusMessagesDiv.innerHTML = '<div class="loader"></div><p>Processing your request... This may take a few moments.</p>';
    
    sentimentResultsDiv.style.display = 'none';
    sentimentContentDiv.innerHTML = '';
    ctaTrendAnalysisContainer.style.display = 'none';

    // Clear previous trend data from session storage
    sessionStorage.removeItem('trendAnalysisData');
    sessionStorage.removeItem('searchQueryForTrend');


    try {
        const response = await fetch('/analyze_youtube_full_pipeline/', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        
        let result;
        try {
            result = await response.json();
        } catch (e) {
            result = { detail: `Failed to parse server response. Status: ${response.status}` };
        }
        
        statusMessagesDiv.innerHTML = '';

        if (response.ok) {
            let sentimentOutputGenerated = false;
            if (result.sentiment_results) {
                let downloadLinksHTML = '';
                let mainSentimentPlotsHTML = '';
                let wordCloudPlotsHTML = '';
                let otherSentimentPlotsHTML = '';

                if(result.sentiment_results.analyzed_data_url) {
                    downloadLinksHTML += `<p><a href="${result.sentiment_results.analyzed_data_url}" class="download-link" target="_blank" download>Download Sentiment Analyzed CSV</a></p>`;
                }

                if (result.sentiment_results.plot_urls && Object.keys(result.sentiment_results.plot_urls).length > 0) {
                    for (const [name_stem, url] of Object.entries(result.sentiment_results.plot_urls)) {
                        const displayName = formatDisplayName(name_stem, "Sentiment");
                        // Each plot, including word clouds, gets its own plot-container for consistency
                        const plotHTML = `<div class="plot-container"><p><strong>${displayName}</strong></p><img src="${url}?cb=${new Date().getTime()}" alt="${displayName}"></div>`;

                        if (name_stem.includes('wordcloud')) {
                            wordCloudPlotsHTML += `<div class="wordcloud-item">${plotHTML}</div>`;
                        } else if (name_stem.includes('distribution') || name_stem.includes('polarity') || name_stem.includes('subjectivity')) {
                            mainSentimentPlotsHTML += plotHTML;
                        } else {
                            otherSentimentPlotsHTML += plotHTML;
                        }
                    }
                }
                
                let finalSentimentHTML = '';
                if (downloadLinksHTML) finalSentimentHTML += downloadLinksHTML;

                if (mainSentimentPlotsHTML || wordCloudPlotsHTML || otherSentimentPlotsHTML) {
                    finalSentimentHTML += '<h3>Sentiment Visualizations:</h3>';
                    if (mainSentimentPlotsHTML) finalSentimentHTML += mainSentimentPlotsHTML;
                    if (wordCloudPlotsHTML) {
                        finalSentimentHTML += `<div class="wordcloud-row">${wordCloudPlotsHTML}</div>`;
                    }
                    if (otherSentimentPlotsHTML) finalSentimentHTML += otherSentimentPlotsHTML;
                } else if (result.sentiment_results.analyzed_data_url) { 
                    finalSentimentHTML += '<p class="info-message">No sentiment plots were generated for this analysis, but sentiment data (CSV) is available for download.</p>';
                }
                
                if (finalSentimentHTML) { 
                    sentimentResultsDiv.style.display = 'block';
                    sentimentContentDiv.innerHTML = finalSentimentHTML;
                    sentimentOutputGenerated = true;
                }
            }

            // Trend Analysis CTA and Data Storage
            if (result.trend_analysis_results && (Object.keys(result.trend_analysis_results.plot_urls || {}).length > 0 || result.trend_analysis_results.detailed_predictions_csv_url || result.trend_analysis_results.cluster_summary_csv_url)) {
                sessionStorage.setItem('trendAnalysisData', JSON.stringify(result.trend_analysis_results));
                sessionStorage.setItem('searchQueryForTrend', data.search_query); // Store query for context on trends page
                ctaTrendAnalysisContainer.style.display = 'block';
                goToTrendAnalysisButton.onclick = () => {
                    window.location.href = 'trends.html';
                };
            } else {
                ctaTrendAnalysisContainer.style.display = 'none';
            }
            
            if (!sentimentOutputGenerated && !result.trend_analysis_results) {
                let messageText = result.message || "Analysis completed. No specific data or plots were generated for the given parameters.";
                let messageClass = "info-message";
                if (result.message && (result.message.toLowerCase().includes("error") || result.message.toLowerCase().includes("failed") || result.message.toLowerCase().includes("no videos found"))) {
                    messageClass = "error-message";
                }
                statusMessagesDiv.innerHTML = `<div class="${messageClass}">${messageText}</div>`;
            } else if (result.message && !(result.message.toLowerCase().includes("success") && (sentimentOutputGenerated || result.trend_analysis_results))) { 
                // Show server message if it's not a generic success when we already have results
                let messageClass = "info-message";
                 if (result.message.toLowerCase().includes("error") || result.message.toLowerCase().includes("failed")) messageClass = "error-message";
                 else if (result.message.toLowerCase().includes("warning")) messageClass = "info-message"; // Or a warning style
                statusMessagesDiv.innerHTML = `<div class="${messageClass}">${result.message}</div>` + statusMessagesDiv.innerHTML;
            }


        } else { 
            statusMessagesDiv.innerHTML = `<div class="error-message"><strong>Error ${response.status}:</strong> ${result.detail || 'An unknown error occurred while processing your request.'}</div>`;
        }
    } catch (error) { 
        statusMessagesDiv.innerHTML = `<div class="error-message"><strong>Network or Client-side Error:</strong> ${error.message || 'A problem occurred while trying to fetch results.'} Please check your network connection and browser console for details.</div>`;
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.removeAttribute('aria-busy');
    }
}

function formatDisplayName(name_stem, typePrefix) { // Kept robust
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
                        .replace('category distribution', 'Category Distribution')
                        .replace('sentiment polarity distribution', 'Sentiment Polarity Distribution')
                        .replace('sentiment subjectivity distribution', 'Sentiment Subjectivity Distribution')
                        .replace('top keywords', 'Top Keywords');
    
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