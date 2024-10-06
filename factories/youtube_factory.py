from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_youtube import youtube_scrape

class YouTubeScraper(Scraper):
    def scrape(self, video_id):
        return youtube_scrape(video_id)

class YouTubeScraperFactory(ScraperFactory):
    def create_scraper(self):
        return YouTubeScraper()
