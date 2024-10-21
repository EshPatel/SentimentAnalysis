from factories.scraper_factory import Scraper, ScraperFactory
from scrapers.scrape_sikayetvar import sikayetvar_scrape

class SikayetVarScraper(Scraper):
    def scrape(self, search_query):
        return sikayetvar_scrape(search_query)

class SikayetVarScraperFactory(ScraperFactory):
    def create_scraper(self):
        return SikayetVarScraper()