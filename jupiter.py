#code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

# creates a Moons class cotaining multiple methods
class Moons:
    
    # initialises the class
    def __init__(self):
        
        # accesses and reads the database, allowing it to be analysed and manipulated
        database_service = "sqlite"
        database = "jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "moons"
        self.data = pd.read_sql(query, connectable)
        
        # changes sets the index of the table to the moon name
        self.data = self.data.set_index("moon")
        
    
    # prints of a statistical summary of the data    
    def summary(self):
        
        return self.data.describe()

    # prints the correlation between each variable in the data        
    def correlations(self):
        

        return self.data.corr()