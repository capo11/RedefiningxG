# Redefining xG
This project aims to **"redefine" the Expected Goal index** (also known as *xG*), allowing to **create an index which takes more information into account when deciding the probability of scoring a goal**.

In this notebook, informations about games and players are taken from different sources, and joined in a dataset, which includes detailed info about **2 seasons of Serie A football (2022/23 and 2023/24 season)**. The dataset includes **player-related features**, like the attacker's **shot rating** and the goalkeeper's **overall rating**, and **contextual features**, like the shot's **body part** or the **goal difference** between the teams at the time of the shot.

This dataset is then used to train a model, using different algorithms, like *XGBoost* or *Random Forest*, or *ensemble methods* like *AdaBoost*. **Data Augmentation** techniques, like *Random Oversampling*, *SMOTE* or *ADASYN* have also been used to test if the model would benefit from having a more balanced dataset.

Other techniques have been used to analyse the model's performance or to improve them, respectively **Cross Validation** (in a *stratified* manner, to keep the folds balanced), and **Parameter Tuning**, to look for a model that could reach a better *f1 score*.

The model's output has been confronted with the xG values produced by a "competitor", *SofaScore*, in order to have a measure of quality for the model's predictions, as **the classic evaluation metrics**, like *precision*, *recall* or *f1 score*, **couldn't reflect the real quality of the model**, since they were based on the binary classifications, instead of the probabilities predicted by the model. For this reason, this notebook tries to reach xG numbers that get close to the competitor one, with **the target being the number of goals scored in the test dataset**.

Eventually, the model has been analysed by looking at **feature importance** by looking at the *Shapley values*, in order to have a global view of the "reasoning" behind the model's predictions, but also to **have a specific explanation of a single prediction**, being able to **understand which features influence a specific prediction for a specific shot**.

A website has been produced to show the results of this model. The results can be seen at [redefiningxg.streamlit.app](https://redefiningxg.streamlit.app/) .
