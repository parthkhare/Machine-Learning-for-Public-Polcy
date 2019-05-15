# Assignment III: Improving Pipeline
[Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline/ML_ImprvPipe_HW3.ipynb)

### Repository Structure
- data: contains Donor's Choose data used for the model 
- charts: contains visualisation of a sample tree 

### Dependencies
+ Dataset: Modified version of Kaggle [Donor's Choose](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data)
+ Primary libaries: sklearn metrics and classifiers, graphviz
+ Helper functions I: Helper function for Preprocessing data & charts
+ Helper functions II: Helper functions on classification metrics, splitting temporal data, loop over classifier models, best model run


### Broad Assumptions
+ Used sample gid size: Increasing grid size might alter performances
	- particularly since KNN took average of 18 hours even under optimal kdtree assumptions
+ Excluded features with high categories: teacher, studentid, geography (/overlaps with school district)
+ Inclusion of other attributes (e.g. demographic from census) might also affect/improve performance
+ With the growth of the program and subsequently more training data one can expect a potential improvement in prediction
