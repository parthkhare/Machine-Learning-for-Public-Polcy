# Assignment V: Re-improving Pipeline
[Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/ML_ImprvPipeII_HW5.ipynb)

### Documents to read
- Policy Audience [Writeup](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/Donor's Choose-Writeup.pdf) 
- [Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/ML_ImprvPipeII_HW5.ipynb)
- [Technical Note](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/TechnicalNote_Guidelines.md): guidelines and subjective judgements taken for the model
	- Subjective calls on features (distrbutions), classifiers and metrics selection 
	- Addresses section for non-policy audience and self reference
	- First hand impressions from data


### Broad Assumptions/Model Limitations
+ Used sample gid size: Increasing grid size might alter performances
	- particularly since KNN took average of 18 hours even under optimal kdtree assumptions
+ Excluded features with high categories: teacher, studentid, geography (/overlaps with school district)
+ Inclusion of other attributes (e.g. demographic from census) might also affect/improve performance
+ With the growth of the program and subsequently more training data one can expect a potential improvement in prediction

### Repository Structure
- data: Donor's Choose data used for the model 
- charts: from the model run
- model summary: 
	- aggregated results from different parameters and classifiers 
	- model timings
- guidelines:
	- note on precision and recall
	- class guidelines on common mistakes


### Dependencies
+ Dataset: Modified version of Kaggle [Donor's Choose](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data)
+ Primary libaries: sklearn classifiers and metrics and, graphviz
+ Helper functions I: Helper function for Preprocessing data & charts
+ Helper functions II: Helper functions on classification metrics, splitting temporal data, loop over classifier models, best model run
+ see requirements.txt for a exhaustive list of modules used

### Notes: Improving Pipeline 
- Last pipeline submitted late due to a health emergency (hence unable improve subjective comments) 
- Implemented all the following changes discussed in extra class:
	+ Dynamic Train-Test split: via feature_eng (function)
	+ 60 days buffer between train and testing periods: via observention window (parameter)
	+ Changed label: 0/1
	+ reduced manual harcoding
	+ Expanded parameters values
	+ Added more useer defined functions


