mport pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
 
class ModelTrainer:
    def _init_(self, X_path, y_path):
        self.X = pickle.load(open(X_path, "rb"))
        self.y = pickle.load(open(y_path, "rb"))
 
    def train(self):
        print("Training models...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        models = {
            "LogisticRegression": LogisticRegression(),
            "SVC": SVC(),
            "RandomForest": RandomForestClassifier()
        }
        os.makedirs("models", exist_ok=True)
        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            metrics[name] = acc
            pickle.dump(model, open(f"models/{name}.pkl", "wb"))
            print(f"{name}: Accuracy = {acc:.4f}")
        pickle.dump(metrics, open("models/metrics.pkl", "wb"))
        return "models/metrics.pkl"
