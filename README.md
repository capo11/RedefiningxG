<h1>Redefining xG</h1>

This project aims to <b>redefine the <i>Expected Goal</i> index</b> (also known as <i>xG</i>), allowing to create an index which takes more information into account when 
deciding the probability of scoring a goal.

This project is part of a <b>Master Thesis</b> (described in the <i>papers</i> folder), which starts from some questions: <b>current xG models may not take into 
consideration factors which could be specific to players and teams</b>, such as the player's skill or the team's level, having an index which describes probabilites
<b>for the average player</b>. The thesis investigates <b>whether introducing player-specific features could bring a significative impact on the predictions</b>. 

Data about games and players are taken from different sources, using <b>different dataset types</b> (from datasets with a couple of features to much more complex datasets) 
and <b>testing different algorithms and techniques</b> (from simple methods as logistic regression to ensemble methods and neural networks). 
Eventually, the majority of data were taken from <i>StatsBomb</i>, retrieving open data, mostly from the 2015/16 season, but data from <i>SoFIFA, 
ClubElo</i> and <i>SofaScore</i> were also useful to retrieve data about player information and team quality.

<h3>Dataset Types</h3>
<ol>
  <li><i>Baseline</i>: simple dataset, only including the shot <b>distance and angle</b></li>
  <li><i>Intermediate</i>: includes <b>player-specific and team related features</b>, as the player and goalkeeper's skill, the teams' Elo Ratings, and more</li>
  <li><i>Full (Improved)</i>: introduces data regarding the other <b>players' positioning and presence</b> with two numerical features, one determining the number of opposite players between the ball and the goal, and one quantifying the minimum distance from the opposite players to the ball.</li>
</ol>

<img width="1069" height="473" alt="image" src="https://github.com/user-attachments/assets/6c3affdf-36d7-4b78-a592-144373f2189f" />

<h3>Tested Algorithms</h3>
<ol>
  <li><i>Logistic Regression</i> (nice results for a simple algorithm)</li>
  <li><i>XGBoost</i> (proved to be the best in terms of stability)</li>
  <li><i>LightGBM</i> (proved to be the best in terms of metrics)</li>
  <li><i>Random Forest</i> (good results, but not enough for this task)</li>
  <li><i>AdaBoost</i> (not suitable for this task)</li>
  <li><i>Neural Networks</i> (good results, but not enough for this task)</li>
</ol>

Other techniques have been used to analyse the model's performance or to improve them, respectively <i>Cross Validation</i> (in a stratified manner, to keep the folds balanced),
and <i>Parameter Tuning</i>, to look for a model that could reach a better f1 score.

The model's output has been compared to the values produced by a "competitor", StatsBomb, in order to have a measure of quality for the model's predictions, 
comparing some evaluation metrics and the xG values produced by the models. By looking at these results, it has been showed how <b>the proposed models reach satisfying results</b>,
close to the StatsBomb performances, having much less training data. On the other hand, <b>the results for aggregated prediction for single players looks to be better
for the proposed models</b>, in some cases, proving how it takes only a handful of information to produce performing models, if the right techniques and the right features
are provided.

<img width="1103" height="367" alt="image" src="https://github.com/user-attachments/assets/a14ddb4e-5ecd-46fe-81ef-4674e4199f5a" />

<br></br>

<b>Model interpretation</b>, implemented by computing <i>Shapley values</i>, has also been useful to help describing how a model reasons, and which features impact predictions
in the different leagues and datasets. These values are useful to describe a model's behaviour, but also to explain the logic behind a prediction for a single shot.


<img width="781" height="940" alt="top5" src="https://github.com/user-attachments/assets/7fdd8b8e-4357-42c1-890e-9df4040b6c26" />
<br></br>
The <i>redefiningxg</i> notebook contains all the steps for this project, plus some usage examples, to store games, obtain predictions, test models and 
explain models and shots.
