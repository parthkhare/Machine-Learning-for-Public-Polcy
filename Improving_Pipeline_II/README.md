# Assignment V: Re-improving Pipeline
[Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/ML_ImprvPipeII_HW5.ipynb)

### Documents to read
- Policy Audience [Writeup](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/Donor's%20Choose-Writeup.pdf) 
- [Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/ML_ImprvPipeII_HW5.ipynb)
- [Technical Note](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline_II/TechnicalNote_Guidelines.md) 
	- Subjective calls on features (distrbutions), classifiers and metrics selection 
	- Addresses section for non-policy audience and self reference
	- First hand impressions from data

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

### Notes: Improving on previous Improving Pipeline Assignment (HW3) 
- Last pipeline submitted late due to a health emergency (hence unable to improve on subjective comments given on assignment hw3) 
- Implemented all the following changes discussed in extra class/notes:
	+ Dynamic Train-Test split: via feature_eng (function)
	+ 60 days buffer between train and testing periods: via observention window (parameter)
	+ Changed label: 0/1
	+ reduced manual harcoding
	+ Expanded parameters values
	+ Added more useer defined functions


