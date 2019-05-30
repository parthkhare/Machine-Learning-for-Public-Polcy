### Refernce guide for Imbalanced Classes: Potential Issues with Accuracy ROC
- ROC curves should be used when there are roughly equal numbers of observations for each class.
- Since ROC curves present an optimistic picture of the model on datasets with a class imbalance.
- ROC curve with an imbalanced dataset might be deceptive and lead to incorrect interpretations of the model skill.

### Metric Selection: Partial Imbalance in the classes -> Metric used F1
- Precision-Recall curves should be used when there is a moderate to large class imbalance.
- e.g. a classifier that tells if a person is honest or not.

#### Precision: measure of result relevancy/ how many selected items are relevant?
- assume that you can misclassify a liar person as honest but not often
- trying to identify liar from honest as a whole group
  - Judicial System

#### Recall: measure of how many truly relevant results are returned/ how relevant items are selected ?
- Focus => coverage/outreach
- really concerned if you think a liar person to be honest
-  it's okay if you classified someone honest as a liar but your model should not claim a liar person as honest
- focusing on a specific class and you are trying not to make a mistake about it.
    - Disease patients  => find all patients who actually have the disease => care if non-infected is diseased or not!
    - Website/Amazon: good chances to be a buyer => don't care about the guy that is not going to buy (so precision is not important)
    
### Precison vs Recall Note
- Precison: Of all the schools that were predicted as recepients of donations, what fraction actually got the funding ?
- Recall: Of all the schools that actually received the funding, what fraction was correctly predicted as recepient of donations ? 

- We care about the schools that did not receive donation within 60 days therefore Recall <> Precision
