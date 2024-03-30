from db.models import TrainingModel
from db import session


class CrudTrainingModel:
    @staticmethod
    def create_training(training):
        session.add(training)
        session.commit()

    @staticmethod
    def get_training(id):
        return session.query(TrainingModel).filter(TrainingModel.id == id).first()

    @staticmethod
    def get_all_trainings():
        return session.query(TrainingModel).all()

    @staticmethod
    def update_training(id, training):
        to_update = session.query(TrainingModel).filter(
            TrainingModel.id == id).first()
        to_update.name = training.name
        to_update.start_time = training.start_time
        to_update.end_time = training.end_time
        to_update.epoch_count = training.epoch_count
        to_update.loss_plot_path = training.loss_plot_path
        to_update.loss_per_epoch = training.loss_per_epoch
        to_update.model_configuration = training.model_configuration
        to_update.experiment_id = training.experiment_id
        session.commit()

    @staticmethod
    def delete_training(id):
        to_delete = session.query(TrainingModel).filter(
            TrainingModel.id == id).first()
        session.delete(to_delete)
        session.commit()
