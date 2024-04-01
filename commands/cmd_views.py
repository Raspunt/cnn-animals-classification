import os
from db.crud.crud_test_model import CrudTestModel
from db.crud.crud_traning_model import CrudTrainingModel
from nn.nn_file_manager import NNFileManager
from nn.nn_trainer import NNTrainer
from nn.nn_tester import NNTester
from settings import settings


class CmdViews():

    def __init__(self) -> None:
        self.nnfm = NNFileManager()
        self.trainer = NNTrainer()
        self.tester = NNTester()

    def clear_test_data(self):
        CrudTestModel.delete_all_test()
        print("All test data has been cleared.")

    def clear_train_data(self):
        CrudTrainingModel.delete_all_training()
        self.nnfm.create_plot_folder()
        self.nnfm.remove_experiments_folder()
        print("All training data has been cleared.")

    def test(self, test_dataset):

        # count_experiments = os.listdir(settings.experiments_folder)
        # for i in range(1, len(count_experiments)):
        #     self.tester.test(test_dataset, i)
        # print("Testing has been completed.")

        test_model = CrudTestModel.get_best_accuracy()
        self.nnfm.save_best_test_model(test_model)

    def train(self, train_dataset, dataset, labels):
        self.trainer.train(train_dataset, dataset, labels)

        print("Training has been completed.")
