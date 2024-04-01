from sqlalchemy import Column,JSON, Integer, String, DateTime, Float, PickleType, ForeignKey
from sqlalchemy.orm import relationship

from db import Base


class TrainingModel(Base):
    __tablename__ = 'TrainingModel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    epoch_count = Column(Integer)
    learning_rate = Column(Float)
    momentum = Column(Float)
    loss_plot_path = Column(String)
    loss_per_epoch = Column(PickleType)
    model_configuration = Column(String)


class TestModel(Base):
    __tablename__ = 'TestModel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    total_predictions = Column(JSON)
    correct_predictions = Column(JSON)
    accuracy = Column(Float)
    nn_model_path = Column(String)
    result_plot_path = Column(String)

