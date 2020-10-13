# deep-learning-for-mental-health
## **Characterization of mental health conditions from texts on social networks** ##

#### Project winner the José Mauro de Castilho award in the 35th edition of the Brazilian Symposium on Databases (SBBD) realized in 2020. #### 
See this article [here](http://sbbd.org.br/2020/wp-content/uploads/sites/13/2020/09/Characterizing-Anxiety-ST7.pdf), and its presentation [here](https://youtu.be/Ftej7HKpbKw?list=PLRKeuVfLlY-5IZme8klDjd0S7I6QWUPQv&t=1841).

This repository documents experiments and results of the proposing to ensemble stacking classifier for the automatic identification of depression, anxiety, and their comorbidity, using a self-diagnosed dataset extracted from Reddit. At the lowest level, binary classifiers developed with deep learning techniques make predictions about specific disorders. A meta-learner explores these weak classifiers as a context for reaching a multi-class, multi-label decision.
![ilustration.png](https://github.com/borbavanessa/deep-learning-for-mental-health/blob/master/images/ilustration.png)

To run the project it is necessary to install an environment containing the packages:

* Python 3.6 or higher and its set of libraries for machine learning (Sckit-learning, NumPy, Pandas)
* Keras 2.2.5
* Tensorflow 1.14.0
* Jupyter Notebook (if you want to run the experiments of the Jupyter files)

The dataset used for developing this model was made available for this work under a data usage contract and, for this reason, is not available with the project.
