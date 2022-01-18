from api.function import model_initial,image_process,predict

# model = model_initial()

def translate(input_path:str):
    model = model_initial()
    processed_img = image_process(input_path)
    output_char = predict(model,processed_img)
    return output_char


