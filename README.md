## A Model for World Cup 2018 Goals Prediction

### Introduction

This project uses the FIFA ranking data and the historical International Soccer matches 
to predict the number of goals each team gets in a game. 

### Datasets:

I use four datasets:

* FIFA rankings from 1993 to 2018 (From [Tadhg Fitzgerald](https://www.kaggle.com/tadhgfitzgerald))
* International Soccer matches from 1872 to 2018 (From [Mart Jürisoo](https://www.kaggle.com/martj42))
* FIFA World Cup 2018 data set (From [Nuggs](https://www.kaggle.com/ahmedelnaggar))
* Countries and Continents Data (From [annexare](https://github.com/annexare/Countries))

In addition, my project is inspired by [Dr. James Bond](https://www.kaggle.com/agostontorok/soccer-world-cup-2018-winner)'s
project on predicting the World Cup 2018 winner. Some codes on feature extractions and data cleaning are from his project.

### Methods
Basically, I use two types of data to predict the number of goals of each team in a game.

* Match Specific Data:
    e.g. whether the team is home team, whether the tournament is world cup, whether the match is importance, etc.

* Team Specific Data:
    1. I extracted which continent each team belongs from the Countries and Continent Dataset. 
       I suppose teams from different continents could have different styles which might affect the final results.
    2. I used the FIFA ranking data to proxy the team's level for each match, this part is originally from [Dr. James Band](https://www.kaggle.com/agostontorok/soccer-world-cup-2018-winner).
    3. I consider the goals from recent matches, e.g. goals on the previous game, moving average of goals in the last 3(or 5, 10, 15, 30) matches.
       Moving averages turn out to be good estimates for goals.

For each match, I constructed two rows of data: one for home team and one for opponent team.
Both rows share the match specific data but not the team specific. In this way, I can double my sample space.

For countries, I only use the labelencoder in Sklearn to encoder countries to integer. 

For models, I tried random forest and an ensemble model combined with random forest, ridge classification and 
gradient boosting classification. 
  
### Results
My prediction accuracy is around 45%. The result for world cup prediction is saved under the "result" folder.

I alse generated a feature importance graph shown below:
![Figure 1](result/Figure_1.png?raw=true)

This graph shows that the most important feature is 3-games moving average for number of goals. 



### Further thoughts about predicting goals
The accuracy can definitely be improved by using different models or by including more data.

For example, the number of goals is clearly a sequential variable. Goals on recent previous matches 
have huge impact on the target match (just as the moving average variables indicate). One could use
a time series model or even a deep learning model (like RNN or LSTM) to predict goals in the next match.

In addition, the accuracy could be improved if we could get player-level data. A team, whether it is national or 
local, consists of 11 players. If we know the performance, such as goals, running distance, etc, we can largely improve
our prediction accuracy.

Another approach could be training machine to watch the football game. If we can feed the machine with videos or even simplified videos
like many betting companies provide, machine could learn directly from these matches instead of learning from some highly summerized statistics.
   
 
