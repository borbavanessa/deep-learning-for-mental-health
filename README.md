# deep-learning-for-mental-health
## **Characterization of mental health conditions from texts on social networks** ##

This repository documents experiments and results of the proposing to ensemble stacking classifier for the automatic identification of depression, anxiety, and their comorbidity, using a self-diagnosed dataset extracted from Reddit. At the lowest level, binary classifiers developed with deep learning techniques make predictions about specific disorders. A meta-learner explores these weak classifiers as a context for reaching a multi-class, multi-label decision.
![ilustration.png](https://github.com/borbavanessa/deep-learning-for-mental-health/blob/master/images/ilustration.png)

To run the project it is necessary to install an environment containing the packages:

* Python 3.6 or higher and its set of libraries for machine learning (Sckit-learning, NumPy, Pandas)
* Keras 2.2.5
* Tensorflow 1.14.0
* Jupyter Notebook (if you want to run the experiments of the Jupyter files)

The dataset used for developing this model was made available for this work under a data usage contract and, for this reason, is not available with the project.
