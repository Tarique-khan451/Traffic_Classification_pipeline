import pickle, os
import numpy as np
 
class DataLoader:
    def _init_(self, X, y):
        self.X = X
        self.y = y
 
    def load(self):
        print("Saving processed data...")
        os.makedirs("data/processed", exist_ok=True)
        pickle.dump(self.X, open("data/processed/X.pkl", "wb"))
        pickle.dump(self.y, open("data/processed/y.pkl", "wb"))
        print("Data saved to data/processed/")
        return "data/processed/X.pkl", "data/processed/y.pkl"
