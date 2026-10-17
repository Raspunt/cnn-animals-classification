from db.crud.crud_test_model import CrudTestModel
from db.crud.crud_traning_model import CrudTrainingModel
from nn.dataset_reader import DatasetReader
from nn.nn_file_manager import NNFileManager
from nn.nn_tester import NNTester
from nn.nn_trainer import NNTrainer


class CmdViews:
    def clear_test_data(self):
        CrudTestModel.delete_all_test()
        print("All test data has been cleared.")

    def clear_train_data(self):
        CrudTrainingModel.delete_all_training()
        NNFileManager().remove_experiments_folder()
        print("Training records and experiments cleared. The best model is preserved.")

    def test(self, model_path=None):
        return NNTester().test(model_path)

    def train(self):
        reader = DatasetReader()
        dataset, labels = reader.read_datasets()
        train_dataset, _ = reader.test_train_dataset(dataset, labels)
        return NNTrainer().train(train_dataset, dataset, labels, reader.classes)
