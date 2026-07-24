
# Add to the end of your training script
import joblib
joblib.dump(best_rf, "model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Saved model.pkl and preprocessor.pkl")
