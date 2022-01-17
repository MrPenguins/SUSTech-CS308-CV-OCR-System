def CreateDataLoader(opt):
    from data.base_data_loader import BaseDataLoader
    data_loader = BaseDataLoader()
    data_loader.initialize(opt)

    return data_loader