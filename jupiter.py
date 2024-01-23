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
    
    def dis_plot(self, variable, k):
        
        # k can change depending on what type of graph is desired
        sns.catplot(data=self.data, x="group", y=variable, kind=k)
        plt.title
        plt.show
   
            
            
    # prints off all data for a specific selected moon
    def specific_moon(self, name):
        

            
        if name in self.data.index:
            # locates the row at which the selected moon is on
            moon_data = self.data.loc[name]
                
            return moon_data
        
        else:
            print("Moon not in database")
    
    # allows for a specific data point to be selected for a specified moon
    def specific_data(self, name, attribute):
        if name in self.data.index:
            moon_data = self.data.loc[name]
            
            # creating a dictionary of each data attribute
            attributes = {"Period": moon_data["period_days"], "Distance": moon_data["distance_km"],
                         "Radius": moon_data["radius_km"], "Magnitude": moon_data["mag"],
                         "Mass": moon_data["mass_kg"], "Group": moon_data["group"],
                         "Eccentricity": moon_data["ecc"], "Inclination": moon_data["inclination_deg"]}
            
            print(f"{name}'s {attribute} is {attributes[attribute]}")
            
            
    def jupiter_mass(self):
        
        # gravitational constant G used in keplers third law
        G =  6.67*(10**-11)

        # creates new columns in the data with the necessary unit conversions for keplers third law
        self.data["t_squared"] = (self.data["period_days"]*24*60*60)**2
        self.data["a_cubed"] = (self.data["distance_km"]*1000)**3
        
        # assigns the two new columns to new variable names
        X = self.data[["a_cubed"]]
        Y = self.data["t_squared"]
        
        #splits 30% of data into testing and the other 70% for training. this is done to see how well the model fits with new data
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        
        # creating a linear regression model
        from sklearn import linear_model
        
        # changing the hyperparemeter so that the line is forced through the origin, as a distance of 0 wpuld cause period to be 0
        model = linear_model.LinearRegression(fit_intercept=False)
        
        # training the data
        model.fit(x_train, y_train)
        # using the test data to make predictions
        pred = model.predict(x_test)
        
        # plotting a cubed against t squared
        sns.relplot(data=self.data, x="a_cubed", y="t_squared")
        
        # plots the predicted regression line
        plt.plot(x_test, pred, color='orange', linewidth = 0.5)
        
        # defining the gradient of the regression model
        gradient = model.coef_[0]
        
        # using the data and rearranging keplers law to find a mass value
        mass = (4*np.pi**2)/(G*gradient)
        
        print(mass)
        
        from sklearn.metrics import r2_score, mean_squared_error
        
        # calculates how well the real data fits the predicted data
        print(f"r2_score is {r2_score(y_test, pred)}")