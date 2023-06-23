from data import BottleDetectorData
from model import BottleDetectorModel
from os import getcwd


model_controller = BottleDetectorModel()
model = model_controller.create_model()
model_controller.compile_model(model)
model.load_weights(getcwd() + r'\checkpoints\my_checkpoint')


data_controller = BottleDetectorData()
training_data, testing_data = data_controller.get_data(getcwd() + r'\dataset')


model.evaluate(testing_data)
