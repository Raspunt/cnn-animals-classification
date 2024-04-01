from nn.dataset_reader import DatasetReader

from db import Base, engine

from commands.make_commands import make_commands
from commands.shemas import CmdCommand
from commands.cmd_views import CmdViews

Base.metadata.create_all(engine)

rd = DatasetReader()
cmd_views = CmdViews()


def nn_init():
    (dataset, labels) = rd.read_datasets()
    (train_dataset, test_dataset) = rd.test_train_dataset(dataset, labels)
    return (train_dataset, test_dataset, dataset, labels)


def run():
    (train_dataset, test_dataset, dataset, labels) = nn_init()

    train_command = CmdCommand("train", lambda: cmd_views.train(
        train_dataset, dataset, labels), "start training of neural network")
    test_command = CmdCommand("test", lambda: cmd_views.test(
        test_dataset), "start test of experiments")

    clear_test_command = CmdCommand(
        'clear-test-data', cmd_views.clear_test_data, 'clear all data from test from database')
    clear_traning_command = CmdCommand(
        'clear-traning-data', cmd_views.clear_test_data, 'clear all data from traning from database and folders')

    commands = [
        train_command,
        test_command,
        clear_test_command,
        clear_traning_command
    ]

    

    make_commands(commands)
