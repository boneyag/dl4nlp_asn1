# dl4nlp_asn1
Deep learning for NLP - Assignment 1

```
dl4nlp_asn1

|--data\
   |--test\ 
   |--train\ 
|--serialized\
|--src\
   |--logistic_reg.py
   |--main.py
   |--pre_data.py   
|--test\
```

Create a serialized dir to store pickle files.
Go to src and run the `main.py` script. ex. `python3 main.py`.

It will first prepare the vocabulary and create data for trainig, validation and testing. 

Once the data is prepared the model will be trained. At the end of the training a graph will be plotted showing the accurayc of training and validation data set in each epoch.

Finally, it will run the test data through the trained logistic regression model and print the test accuracy of the model.
