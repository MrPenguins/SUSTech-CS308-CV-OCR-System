from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--test_model', type=str, default="./checkpoints/stage2_10_one_52000.pkl",
                                 help='select the model you want to load')


        self.isTrain = False
