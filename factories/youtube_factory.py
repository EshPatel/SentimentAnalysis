from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_youtube import youtube_scrape

class YouTubeScraper(Scraper):
    def scrape(self, search_query, video_limit = None):
        return youtube_scrape(search_query, video_limit)

class YouTubeScraperFactory(ScraperFactory):
    def create_scraper(self):
        return YouTubeScraper()
