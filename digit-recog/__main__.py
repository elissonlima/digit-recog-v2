import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG, 
            format='%(asctime)s - %(message)s', 
            datefmt='%d-%m-%Y %H:%M:%S',
            handlers=[logging.StreamHandler()])

TRAIN_DATASET_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),"data","train.csv")
TRAIN_DATASET_NPY_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),"data","train_dataset.npy")
TRAIN_LABEL_NPY_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),"data","train_label.npy")
TEST_DATASET_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),"data","test.csv")
TEST_DATASET_NPY_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),"data","test_dataset.npy")

def load_train_dataset_from_csv():
    train_dataset = np.array([])
    train_labels = np.array([])
    with open(TRAIN_DATASET_CSV_PATH) as csv_train:
        next(csv_train)
        for line in csv_train:
            arr_line = np.array(line.strip().split(",")).astype(np.uint8)
            train_dataset = np.append(train_dataset, arr_line[1:])
            train_labels = np.append(train_labels, arr_line[0])
        return (train_dataset.reshape((train_labels.shape[0],
                28,28)), train_labels)

def load_test_dataset_from_csv():
    test_dataset = np.array([])
    with open(TEST_DATASET_CSV_PATH) as csv_test:
        next(csv_test)
        qtd_lines=0
        for line in csv_test:
            arr_line = np.array(line.strip().split(",")).astype(np.uint8)
            test_dataset = np.append(test_dataset, arr_line)
            qtd_lines+=1
        return test_dataset.reshape((qtd_lines,28,28))

def get_datasets():
    train_dataset = np.array([])
    train_labels = np.array([])
    test_dataset = np.array([])

    logging.info("Getting datasets")
    logging.info("Getting train dataset")
    if (not os.path.isfile(TRAIN_DATASET_NPY_PATH)
        or not os.path.isfile(TRAIN_LABEL_NPY_PATH)):
        logging.warning("Serialized objects does not exists, loading from csv...")
        logging.warning("It should take a while...")
        train_dataset, train_labels = load_train_dataset_from_csv()
        logging.info("DONE")
        np.save(TRAIN_DATASET_NPY_PATH, train_dataset)
        np.save(TRAIN_LABEL_NPY_PATH, train_labels)
        logging.info("Saving train datasets at {}".format(os.path.dirname(TRAIN_DATASET_NPY_PATH)))
    else:
        logging.info("Loading from serialized files at {}".format(os.path.dirname(TRAIN_DATASET_NPY_PATH)))
        train_dataset = np.load(TRAIN_DATASET_NPY_PATH)
        train_labels = np.load(TRAIN_LABEL_NPY_PATH)
    
    logging.info("Getting test dataset")
    if not os.path.isfile(TEST_DATASET_NPY_PATH):
        logging.warning("Serialized objects does not exists, loading from csv...")
        logging.warning("It should take a while...")
        test_dataset = load_test_dataset_from_csv()
        logging.info("DONE")
        np.save(TEST_DATASET_NPY_PATH, test_dataset)
        logging.info("Saving test dataset at {}".format(os.path.dirname(TEST_DATASET_NPY_PATH)))
    else:
        logging.info("Loading from serialized files at {}".format(os.path.dirname(TRAIN_DATASET_NPY_PATH)))
        test_dataset = np.load(TEST_DATASET_NPY_PATH)

    return train_dataset, train_labels, test_dataset


if __name__ == "__main__":
    logging.info("STARTING digit-recog model")
    train_dataset, train_labels, test_dataset = get_datasets()