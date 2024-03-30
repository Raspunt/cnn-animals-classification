from db.models import TrainingModel, Test, Experiment
from db import session


class CrudExperimentModel:
    @staticmethod
    def create_experiment(experiment):
        session.add(experiment)
        session.commit()

    @staticmethod
    def get_experiment(id):
        return session.query(Experiment).filter(Experiment.id == id).first()

    @staticmethod
    def get_all_experiments():
        return session.query(Experiment).all()

    @staticmethod
    def update_experiment(id, experiment):
        to_update = session.query(Experiment).filter(
            Experiment.id == id).first()
        to_update.name = experiment.name
        to_update.pytorch_version = experiment.pytorch_version
        session.commit()

    @staticmethod
    def delete_experiment(id):
        to_delete = session.query(Experiment).filter(
            Experiment.id == id).first()
        session.delete(to_delete)
        session.commit()
