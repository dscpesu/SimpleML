## DRY BEANS CLASSIFICATION

**GOAL**

The goal of this project is to classify dry beans according to the respective categories.

Dataset can be downloaded from [here](https://www.kaggle.com/josegarban/beans-classification)

**WHAT I HAD DONE**

- Understood the data and Discussed some major columns on which dry beans depends
- Handled outliers of diagnosis columns as it's very important to classify the dry beans
- Then I applied different classification models present in sklearn to train the model
- Used Correlation coefficients to measure how strong a relationship is between two variables

**MODELS USED**

-  Logistic Regression
-  Decision Tree
-  K Nearest Neighbour
-  Random Forest Classifier

**RESULTS**

By using Logistic Regression I got 
 ```
    Accuracy of training data: 28.655400440852315
    Accuracy of testing data: 27.889324191968655
 ``` 

 By using Decision Tree I got 
 ```
    Accuracy of training data: 100.0
    Accuracy of testing data: 89.49559255631733
 ``` 

 By using K Nearest Neighbour I got 
 ```
    Accuracy of training data: 86.82691298415031
    Accuracy of testing data: 78.20763956904995
 ``` 

 By using Random Forest Classifier I got 
 ```
    Accuracy of training data: 100.0
    Accuracy of testing data: 92.85014691478942
 ``` 

As the accuracy of **Random Forest Classifier** algorithm is more ie. **92.85%** ~ **93%**. Hence this model is selected.

![accuracy](https://github.com/prathimacode-hub/SimpleML/assets/74645302/00e49738-6c30-4c64-b1da-5b161158bea2)
