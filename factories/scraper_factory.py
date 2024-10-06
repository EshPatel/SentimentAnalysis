from abc import ABC, abstractmethod

class Scraper(ABC):
    @abstractmethod
    def scrape(self, search_query):
        pass

class ScraperFactory(ABC):
    @abstractmethod
    def create_scraper(self):
        pass
