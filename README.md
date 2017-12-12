# Dialect identification using Siamese network
Siamese network based dimensionality reduction for robust dialect identification (and as well as language recognition)

# Requirment
* Python, tested on 2.7.6 (better with jupyter notebook)
* Tensorflow

# Training Model
* train_ivector.ipynb : Training siamese neural network model for i-vector feature
* train_word.ipynb : Training siamese neural network model for word feature
* train_char.ipynb : Training siamese neural network model for character feature
* train_phone.ipynb : Training siamese neural network model for phoneme feature

# Performance evaluation example of i-vector feature on MGB-3 Test dataset

* Confusion matrix

| |EGY|GLF|LAB|MSA|NOR|
|-|-|-|-|-|-|
 |EGY|  225|   13|  37 |  11 |   24|
 |GLF|   11|  178|   31|   16|   12|
 |LAB|   43|   45|  212|   14|   31|
 |MSA|    5|    5|   10|  208|   10|
 |NOR|   18|    9|   44|   13|  267|

* Precision

|EGY|GLF|LAB|MSA|NOR|
|-|-|-|-|-|
|0.73|0.72|0.61|0.87|0.76|

* Recall

|EGY|GLF|LAB|MSA|NOR|
|-|-|-|-|-|
|0.75|0.71|0.63|0.79|0.78|

***Overall performance***

| Accurary  : 0.731| Precision : 0.739| Recall    : 0.732|
|-|-|-|



