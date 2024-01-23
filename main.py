from data_loading import load_and_process_data
from model import create_model
from training import train_model
from evaluation import evaluate_model, plot_results
from constants import DATA_PATH, LEARNING_RATE, BATCH_SIZE, TRAIN_SIZE, RANDOM_STATE

def main():
    X_train, X_test, y_train, y_test = load_and_process_data(DATA_PATH, TRAIN_SIZE, RANDOM_STATE)
    model = create_model(X_train.shape[1])
    trained_model, history = train_model(model, X_train, y_train, X_test, y_test, './', 40, LEARNING_RATE, BATCH_SIZE)
    plot_results(history)
    evaluate_model(trained_model, X_test, y_test)

if __name__ == "__main__":
    main()
