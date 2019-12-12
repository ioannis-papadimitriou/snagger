From https://urbansounddataset.weebly.com/urbansound8k.html:

"This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, 
car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. 
The classes are drawn from the urban sound taxonomy. All excerpts are taken from field recordings uploaded
to www.freesound.org. The files are pre-sorted into ten folds (folders named fold1-fold10)."

"10-fold cross validation using the predefined folds: train on data from 9 of the 10 predefined folds and test 
on data from the remaining fold. Repeat this process 10 times (each time using a different set of 9 out of the 10 
folds for training and the remaining fold for testing). Finally report the average classification accuracy over all 
10 experiments (as an average score + standard deviation, or, even better, as a boxplot)."

Using the predefined folds, a variety of different algorithms were used for classification:

- Random Forest
- SVM
- Logistic Regression
- K-NN
- ANN
- CNN (1D)
- CRNN (Convolutional + Recurrent)
- CNN (2D)
- Autoencoders

The best average performance was achieved by CNN (2D) (73 %, stdev:0.03).

Note: If you are working on this bear in mind that data leakage is guarranteed if you reshuffle and/or merge the data. 
I have seen a lot of ridiculously high accuracies when all the data are used together and split randomly. One can 
reach accuracies higher than 95 %, quickly and easily, with a simple random forest in that case, no deep learning needed.

TODO: Write a small report on this (and maybe try a deep CNN (2D)).

