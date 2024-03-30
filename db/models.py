from sqlalchemy import Column, Integer, String, DateTime, Float, PickleType, ForeignKey
from sqlalchemy.orm import relationship

from db import Base


class TrainingModel(Base):
    __tablename__ = 'TrainingModel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    epoch_count = Column(Integer)
    loss_plot_path = Column(String)
    loss_per_epoch = Column(PickleType)
    model_configuration = Column(String)
    experiment_id = Column(Integer, ForeignKey('ExperimentModel.id'))


class TestModel(Base):
    __tablename__ = 'TestModel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_total_predictions = Column(Integer)
    test_correct_predictions = Column(Integer)
    test_accuracy = Column(Float)
    experiment_id = Column(Integer, ForeignKey('ExperimentModel.id'))


class ExperimentModel(Base):
    __tablename__ = 'ExperimentModel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    model_name = Column(String)
    training = relationship("TrainingModel", uselist=False, backref="experiment")
    test = relationship("TestModel", uselist=False, backref="experiment")
    pytorch_version = Column(String)
