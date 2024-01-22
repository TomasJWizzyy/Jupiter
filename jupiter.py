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
    
    
    # allows for each variable to be plotted against each other. allows for them to be selected as required.        
    def scatter_plot(self, X, Y):
        

        # checks if the input is correct    
        if X in self.data.columns and Y in self.data.columns:
            
            # plots the desired graph depending on user input
            sns.scatterplot(x=X, y=Y, data=self.data)
            plt.title(f"{Y} against {X}")
            plt.xlabel(X)
            plt.ylabel(Y)
                
            plt.show()
                
        else:
            print("One of the chosen variables does not exist")
            
            
    # plots the distribution between groups for any selected variable
    def dis_plot(self, variable, group, separate):
        

            
        if variable in self.data.columns and group in self.data.columns:
            
            # the graph can either be plotted seperatley for each group or into one graph
            if separate == "yes":
                
                sns.displot(x=variable, data=self.data, hue=group, multiple="stack", col=group, col_wrap=3, height=4)
                
                plt.xlabel(variable)
                
                
                plt.show()
                
            elif separate == "no":
                    
                sns.displot(x=variable, data=self.data, hue=group, multiple="stack")
                
                plt.xlabel(variable)
                
                
                plt.show()
                            
            else:
                print("Please enter 'yes' or 'no'")
                
        else:
            print("One of the chosen variables does not exist")