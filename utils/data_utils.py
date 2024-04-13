from enum import Enum
from datetime import datetime

class Season(Enum):
    Past10 = datetime(2013, 8, 1)
    Past7 = datetime(2017, 8, 1)
    Past6 = datetime(2018, 8, 1)
    Past5 = datetime(2019, 8, 1)
    Past2 = datetime(2022, 8, 1)
    Past1 = datetime(2023, 8, 1)

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __ne__(self, other):
        return self.name != other.name
    
    def __lt__(self, other):
        return self.name < other.name
    
    def __le__(self, other):
        return self.name <= other.name
    
    def __gt__(self, other):
        return self.name > other.name
    
    def __ge__(self, other):
        return self.name >= other.name