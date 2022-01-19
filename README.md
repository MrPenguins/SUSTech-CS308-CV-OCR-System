# Sustech-CS308-CV-OCR-System

## The two stage OCR system

Optical Character Recognition, or OCR, is a classical task in computer vision. This task requires converting images of printed or handwritten text into machine-encoded text.

In this project, we implemented a two-stage OCR system. Our OCR system can recognize English uppercase and lowercase characters, and then output the corresponding English text in the picture

- In the first stage, we used the traditional computer vision method (without neural network) to complete Letter Segmentation
- In the second stage, we used the convolutional neural network for letter recognition.
- We generated our own dataset for training and testing.

The **input** is an image of English text, and through the OCR system, we will **output** a machine-encoded text representing the corresponding character information in the image.


## Prerequisites
- Linux or macOS or Windows
- Python 3.6 or more
- Only CPU is OK for training
- Install PyTorch and dependencies from http://pytorch.org
- Other packages
```bash
pip install python_Levenshtein
pip install opencv-python
pip install pillow
```
- Clone this repo:
```bash
git clone https://github.com/MrPenguins/SUSTech-CS308-CV-OCR-System.git
cd SUSTech-CS308-CV-OCR-System
```


### Testing
- A few example test images are included in the `tmp` folder.

- A few example test txt files are included in the `test_txt` folder.

- We upload some trained model, you can directly use them in the `checkpoints` folder.

- After finishing setting the parameter and pkl file, you can do the test now

  `run the command -> python main.py `

  The test results will be directly print to the console.

### Datasets

#### Using existing dataset

We uploaded a dataset of English letters that you can use directly at folder`emnist-chars74k_datasets`

#### Using  you own dataset

If you want to use your own datasets , please follow the following format to create the dataset

```python
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
```

Also, you need to name your training dataset as `Train_png`,  testing dataset as `Test_png`


### Training
- You need to set hyper-parameter before train you model.
```python
('--name', type=str, default="stage2",help='the name of your model')
('--epoch', type=int, default=10,help='the number of epochs')
('--batch_size', type=int, default=128, help='the size of the batch_size')
('--lr', type=int, default=1e-4, help='the learning rate')
('--model', type=str, default="stage2", help='choose the model you want to use')
('--train_data_path', type=str, default="./datasets/Train_png", help='the train data path')
('--test_data_path', type=str, default="./datasets/Test_png", help='the test data path')
```
- After finishing setting the hyper-parameters,  you can train your model now

`python train.py --name model --epoch 10 --train_data_path ./Train_png/ --test_data_path ./Test_png/`

- To view training results, please check the console's out
- To view your `.pkl` file, please check the `checkpoints` directory

## More Training/Test Details
- For more details, you can see our paper named The Two Stage OCR System at the root directory.

## Conclusion

###  A. Challenges and Methods

- Challenge 1: The segmentation of Stage 1 is not accurate, letters that are segmented may not be complete.
- Method 1: Relax the selection of letter boundaries, expand the previous areas so that letters can be fully included.
- Challenge 2: If we directly use the result of Stage 1 to do the letter recognition, the accuracy will not meet our expectations.
- Method 2: First padding the resulting image of Stage 1 and make it has the same width and height. The padding size will be calculated based on the larger values in its height and width, plus some base values. Then use the outcome to do the letter recognition, which can increase accuracy.
- Challenge 3: The model trained by our own dataset does not have a generalization ability. It performs badly on real-world data.
- Method 3: Adding more randomness to our own dataset. The data are not supposed to be in the center of the image always. The letter size and the image size can also be varying.

### B. Advantages and Disadvantages

#### 1)  Advantages of our Approach

Our OCR system is light-weighted, so it can be deployed in different platforms even with weak data processing ability. Our OCR system also has quick infer time so it can be used for real-time letter recognition. Moreover, Stage 1 of our OCR system is not limited to segment English letters. It can also split French letters, the Arabic alphabet, and even Chinese Characters. It can combine with other recognition models.

#### 2) Disadvantages of our Approach

The accuracy of our OCR system is not very high. It will both segment letters wrong in Stage 1 and recognition letters wrong in Stage 2. The generalization ability of our OCR system is also not very good, it may perform badly on unseen real-world data.

### C. Limitations

Our OCR system can only work well on a well-defined input class. The input data should be screenshot-like printed letter images and there should only be English uppercase and lowercase characters in it. The system can not recognize punctuation so users are required to remove them before recognition, or the accuracy may decrease.

### D. Possible Improvement Directions

For Stage 1, since the current approach can not split letters with a very little distance well, we can use a neural network to complete the letter segmentation. For Stage 2, we are going to modify our models so that they can classify punctuation as well. Moreover, we will add support for more languages besides English.



