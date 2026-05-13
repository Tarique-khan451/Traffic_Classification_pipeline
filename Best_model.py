import pickle, os
 
class BestModelSelector:
    def _init_(self, metrics_path):
        self.metrics = pickle.load(open(metrics_path, "rb"))
 
    def select(self):
        print("Selecting best model...")
        best_name = max(self.metrics, key=self.metrics.get)
        best_acc = self.metrics[best_name]
        print(f"Best Model: {best_name} with Accuracy: {best_acc:.4f}")
        best_model = pickle.load(open(f"models/{best_name}.pkl", "rb"))
        os.makedirs("best_model", exist_ok=True)
        pickle.dump(best_model, open("best_model/best_model.pkl", "wb"))
        print("Best model saved to best_model/best_model.pkl")
        return best_model
