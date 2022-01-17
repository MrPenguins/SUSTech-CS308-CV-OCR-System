
def create_model(opt,label_num):
    if opt.model == 'stage2':
        from .stage2_model import stage2Model
        if opt.isTrain:
            model = stage2Model() # train 运行这一步
        else:
            model = stage2Model()

    model.initialize(label_num)

    return model
