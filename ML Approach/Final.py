# Importing the necessary classes and functions from the model script
from RFR_final import run_mainrfr
from KNN_final import run_mainknn
from cnn_final import run_maincnn

def run_all_models():
    # Call the run_main function from the model script
    print("Random Forest: \n")
    run_mainrfr()
    print("KNN: \n")
    run_mainknn()
    print("CNN: \n")
    run_maincnn()

if __name__ == "__main__":
    # Call the function to run all models
    run_all_models()
