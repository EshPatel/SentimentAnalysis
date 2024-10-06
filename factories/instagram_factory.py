from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_instagram import instagram_scrape

class InstagramScraper(Scraper):
    def scrape(self, search_query):
        return instagram_scrape(search_query)

class InstagramScraperFactory(ScraperFactory):
    def create_scraper(self):
        return InstagramScraper()
