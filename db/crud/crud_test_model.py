from sqlalchemy import func

from db.models import TestModel
from db import session


class CrudTestModel:
    @staticmethod
    def create_test(test):
        session.add(test)
        session.commit()

    @staticmethod
    def get_test(id):
        return session.query(TestModel).filter(TestModel.id == id).first()

    @staticmethod
    def get_all_tests():
        return session.query(TestModel).all()

    @staticmethod
    def update_test(id, test):
        to_update = session.query(TestModel).filter(TestModel.id == id).first()
        to_update.test_total_predictions = test.test_total_predictions
        to_update.test_correct_predictions = test.test_correct_predictions
        to_update.test_accuracy = test.test_accuracy
        to_update.experiment_id = test.experiment_id
        session.commit()

    @staticmethod
    def delete_test(id):
        to_delete = session.query(TestModel).filter(TestModel.id == id).first()
        session.delete(to_delete)
        session.commit()

    @staticmethod
    def delete_all_test():
        session.query(TestModel).delete()
        session.commit()

    @staticmethod
    def get_best_accuracy() -> TestModel:
        stmt = session.query(func.max(TestModel.accuracy))
        max_accuracy = session.execute(stmt).scalar()
        return session.query(TestModel).filter(TestModel.accuracy == max_accuracy).first()

        
