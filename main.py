
from collections import Counter

from nn import prepare_optimizer

from nn.NNTrainer import NNTrainer
from nn.DatasetReader import DatasetReader



rd = DatasetReader()

(criterion, optimizer) = prepare_optimizer()
trainer = NNTrainer(optimizer=optimizer, criterion=criterion)


if __name__ == "__main__":

    (dataset, labels, class_to_idx) = rd.read_datasets()
    (train_dataset, test_dataset) = rd.test_train_dataset(dataset, labels)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    class_labels = [idx_to_class[label] for label in labels]
    class_counts = Counter(class_labels)

    print('count of classes')
    print(class_counts)

    trainer.train(train_dataset, dataset, labels)
    # test(test_dataset)
    # start_webcam(idx_to_class)
