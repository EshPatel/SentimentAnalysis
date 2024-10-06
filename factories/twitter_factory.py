from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_twitter import twitter_scrape

class TwitterScraper(Scraper):
    async def scrape(self, search_query):
        return await twitter_scrape(search_query)

class TwitterScraperFactory(ScraperFactory):
    def create_scraper(self):
        return TwitterScraper()
