import pickle

# 1. Saving your trained object to a file
# 'wb' means Write Bytes mode
with open("my_ml_model.pkl", "wb") as file:
    pickle.dump(trained_model, file)

# 2. Loading your trained object back later
# 'rb' means Read Bytes mode
with open("my_ml_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
