from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--train_data_path', type=str, default="./emnist-chars74k_datasets/Train_png", help='the train data path')
        self.parser.add_argument('--test_data_path', type=str, default="./emnist-chars74k_datasets/Test_png", help='the test data path')

        self.isTrain = True
