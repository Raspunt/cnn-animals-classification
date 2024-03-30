from db.models import TrainingModel, Test, Experiment
from db import session


class CrudTestModel:
    @staticmethod
    def create_test(test):
        session.add(test)
        session.commit()

    @staticmethod
    def get_test(id):
        return session.query(Test).filter(Test.id == id).first()

    @staticmethod
    def get_all_tests():
        return session.query(Test).all()

    @staticmethod
    def update_test(id, test):
        to_update = session.query(Test).filter(Test.id == id).first()
        to_update.test_total_predictions = test.test_total_predictions
        to_update.test_correct_predictions = test.test_correct_predictions
        to_update.test_accuracy = test.test_accuracy
        to_update.experiment_id = test.experiment_id
        session.commit()

    @staticmethod
    def delete_test(id):
        to_delete = session.query(Test).filter(Test.id == id).first()
        session.delete(to_delete)
        session.commit()
