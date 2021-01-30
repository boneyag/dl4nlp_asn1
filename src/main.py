import pickle

from prep_data import prepare_vocabulary
from prep_data import prepare_data
from logistic_reg import plot
from logistic_reg import training
from logistic_reg import test_model
from logistic_reg import plot

def main():
    prepare_vocabulary()
    prepare_data()

    vocab = pickle.load(open("../serialized/vocab3.pkl", "rb"))
    

    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open("../serialized/data3.pkl", "rb"))
    # model = training(X_train, y_train, X_val, y_val)

    # pickle.dump(model, open("../serialized/log_model3.pkl", "wb"))

    # [tr_ac, vl_ac] = pickle.load(open("../serialized/plot3.pkl", "rb"))
    # plot(tr_ac, vl_ac)

    test_model(X_test, y_test)

if __name__ == "__main__":
    main()