from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_reddit import reddit_scrape

class RedditScraper(Scraper):
    def scrape(self, search_query):
        return reddit_scrape(search_query)

class RedditScraperFactory(ScraperFactory):
    def create_scraper(self):
        return RedditScraper()
