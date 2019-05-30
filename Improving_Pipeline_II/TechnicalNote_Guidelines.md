# Assignment V: Improving Pipeline
[Jupyter Notebook](https://github.com/parthkhare/Machine-Learning-for-Public-Polcy/blob/master/Improving_Pipeline/ML_ImprvPipe_HW3.ipynb)

### The following documents summarises subjective calls taken on features, parameters, evaluation metrics and user-defined functions. It features details not addressed in the policy writeup.


## Data: First Impressions 
Following quick observation may be drawn for data on 124 thousand project proposal over 2 years.
- Multi-modal Dononation response time stands out 
    - particularly the drop in number of donations for around 60days
    - this implies that most of funding were done within this window and therefore we can expect a baseline in our model
- Relatively higher contribution in supplies & technology
    - relatively higher overall donation in urban areas
    - higher donation made in high poverty schools
    - higher donations made in pre-K2 grades
- Total pricing and students reached attributes show a near *Gamma* distribution
    - which *supports* trying a non-parametric/Machine learning based method
    - conventional regression frameworks (in some cases even GLMM) might be inept in accurately extracting variance in such cases
    - pairwise correlation between them can be indicative of more the students reached more donation made 
- Average Monhtly Funding (with support) aorund $400-$500, without much seasonal variation. 
- Sizebale outliers, particular to the range of over $1500K in November 2013


## Features
### Missing Values
- NA types: checked for na, null, NA ,NaN :0/0 forms, '.', different contingent on data type
- Continuous variables: Mean fitting works
	- If the distribution is heavily skewed, like gamma or one needs to be careful
	- Unless we know the context, mean subsitution is not ideal
	- e.g. In case of income variable, we know of presence of inequality, hence it is fine but says students reached, it is better idea to check the distribution first (in present case the missing % were low)
- Categorical/Factor variables:
	- Before imputing by mode category check relative frequencies of 2nd, 3rd highest categories
	- We would not want to bias the estimates by using disproportionately high category
59 missing values student_reached treated by mean replacement.  Ideally, given the right skewed distribution mean is not recommended, but the missing % is less than 0.001% of data. Remaining missing low frequency in categorical variable replaced by median. Not imputing as of now, can use hot deck or association rules if required

### Feature Adjustment to model
- Creating binary 0,1 features from True/False
- Normalising continuous variables
- Dummy feature generation for model run

### Engeneering: Feature Generation & Deselection
- Feature Generation I: using from prefix
    - Gender variable on teachers
    - Considered using selecting variables by qualifications: but techers with Dr's less than 1% of data
- Feature Generation II: major cities
    - Considered high frequency cities
    - Quick google highest frequency ones for having high population
- Feature Deselection: Excluding hierarchial variable -> Focus Area vs Subject (remove focus_area)
    - Checking if focus area and focus subject have 1-1 mapping
    - No they do not
    - Retain category with higher resolution
- Hierarchial geographies: geographical identifiers (lat, long, state, county, city)
    - Multiple geographical indicators indicating same geography
    - Choosing geographical resolution
        - Proportionate to the data density choose higher resolution (but lower than lat,long)
        - Corresponding number of cities (5955) by school district (5970)
        - But given difference in rel frequency using both
        - Keeping School district as it might be a better policy indicative

### Outliers 
No strong pairwise correlation leading to a multinollinearity like situation.There might be correlation between total price and students reached, if outliers are removed, but considering the outliers value as part of sample for now. 
NOTE: *Treating outliers did not seem particularly useful for the data in question. Outliers in donation and students are plausible. If however required Z-scaores are a good way to examine individual feature distirbutions.*



## Classifiers
### Check for feature interaction: poverty
- Ran slightly modified version of the test grid to:
	- check for model running
	- get a quick sense of local optimas of the classifiers

### Rough Cut: TestGrid 
- Checked if class imbalance happened actually exists across different temporal splits 
- LR and KNN: low precision as well as timetaking
    - Call for round II: Drop LR and KNN 
- Narrow spead in GBM could be indicative of overfitting with limited paramters: this gives guidance that one must think overfitting control 
    - Call for round II: preferrence of AB over GB 
    - Increase shrinkage, learning rate as it already seems to overfit
- RF lesser noisier than DT
    - Call for round II: drop DT use RF
- Judgement: try both bagging and boosted ensemble

### Expanded Grid & Best Model selection
- Assess model via AUC-ROC spread across models
- Check precision ranges
- Pruning trees via restricting observation in nodes
- Gradient boosting shows promising signs however vulnerable to overfitting, so tested with stricter controls on parameters
- Gradient boosted model with 100 estimators, learning rate: 0.5 , max depth: 20, min sample split: 500 & subsample 0.8
	- learning rate: 0.5 not kept much lower to avoid mimicking the distribution exactly
	- max depth: 20 and min sample split: 100 both again trying to ensure that fit is close but not exact to work with different test datasets
	- subsample: 0.8, considering < 1 leads to stochastic gradient boosting, this was done to induce more randomness
- Checked best model & fit parameters across for different intertemporal horizons

## Evaluation
- We want to identify which 5% of school projects should a donor prioritize their intervention on
- Checked performance of different models by checking AUC_ROC spread of model and precision  
- Checked for degree of class imbalance and also that exists actoss across different temporal windows
- Selected Precision therefore as the metric of evaluation


## Ancillary functions: In additon to modifying magicloop
- feature_eng: club together various feature transformation processes and allow for dynamic/sequential training and test variable generation 
- compare_metric: 
- temporal split: splitting the data by specified time
- clf_wrapper: integrates the clf_loop across for sliding time windows
- best_model: automates selection of best model based on specified metrics and helps to check it best model's robustness across time periods 
- modified loop function where scope
- pr_curve: plots precision/recall for a limited sample of data 





