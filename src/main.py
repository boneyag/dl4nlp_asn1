import pickle

from prep_data import prepare_vocabulary
from prep_data import prepare_data
from logistic_reg import plot
from logistic_reg import training
from logistic_reg import test_model

def main():
    # prepare_vocabulary()
    # prepare_data()
    # plot()

    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open("../serialized/data.pkl", "rb"))
    # model = training(X_train, y_train, X_val, y_val)

    # pickle.dump(model, open("../serialized/log_model2.pkl", "wb"))

    test_model(X_test, y_test)

if __name__ == "__main__":
    main()