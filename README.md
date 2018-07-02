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

My prediction for all games at the group stage:

| scoreA         | teamA          | teamB          | scoreB | correct_win | correct_score | 
|----------------|----------------|----------------|--------|-------------|---------------| 
| 1              | Russia         | Saudi Arabia   | 2      | no          | no            | 
| 0              | Egypt          | Saudi Arabia   | 2      | yes         | yes           | 
| 3              | Uruguay        | Saudi Arabia   | 0      | yes         | no            | 
| 1              | Russia         | Egypt          | 0      | yes         | no            | 
| 3              | Uruguay        | Egypt          | 0      | yes         | no            | 
| 0              | Russia         | Uruguay        | 3      | yes         | yes           | 
| 1              | Portugal       | Spain          | 1      | yes         | no            | 
| 1              | Morocco        | Spain          | 2      | no          | no            | 
| 0              | Iran           | Spain          | 2      | yes         | no            | 
| 1              | Portugal       | Morocco        | 1      | no          | no            | 
| 1              | Iran           | Morocco        | 1      | no          | no            | 
| 1              | Portugal       | Iran           | 0      | no          | no            | 
| 2              | France         | Australia      | 1      | yes         | yes           | 
| 0              | Peru           | Australia      | 1      | no          | no            | 
| 2              | Denmark        | Australia      | 1      | no          | no            | 
| 2              | France         | Peru           | 0      | yes         | no            | 
| 1              | Denmark        | Peru           | 0      | yes         | yes           | 
| 1              | France         | Denmark        | 1      | yes         | no            | 
| 1              | Argentina      | Iceland        | 1      | yes         | yes           | 
| 1              | Croatia        | Iceland        | 1      | no          | no            | 
| 0              | Nigeria        | Iceland        | 2      | no          | no            | 
| 1              | Argentina      | Croatia        | 1      | no          | no            | 
| 0              | Nigeria        | Croatia        | 2      | yes         | yes           | 
| 1              | Argentina      | Nigeria        | 0      | yes         | no            | 
| 2              | Brazil         | Switzerland    | 0      | no          | no            | 
| 0              | Costa Rica     | Switzerland    | 0      | yes         | no            | 
| 1              | Serbia         | Switzerland    | 1      | no          | no            | 
| 2              | Brazil         | Costa Rica     | 0      | yes         | yes           | 
| 0              | Serbia         | Costa Rica     | 0      | no          | no            | 
| 2              | Brazil         | Serbia         | 0      | yes         | yes           | 
| 1              | Germany        | Mexico         | 0      | no          | no            | 
| 1              | Sweden         | Mexico         | 0      | yes         | no            | 
| 1              | Korea Republic | Mexico         | 0      | no          | no            | 
| 1              | Germany        | Sweden         | 0      | yes         | no            | 
| 1              | Korea Republic | Sweden         | 2      | yes         | no            | 
| 2              | Germany        | Korea Republic | 1      | no          | no            | 
| 1              | Belgium        | Panama         | 0      | yes         | no            | 
| 2              | Tunisia        | Panama         | 0      | yes         | no            | 
| 1              | England        | Panama         | 0      | yes         | no            | 
| 2              | Belgium        | Tunisia        | 2      | no          | no            | 
| 1              | England        | Tunisia        | 2      | no          | no            | 
| 1              | Belgium        | England        | 1      | no          | no            | 
| 0              | Poland         | Senegal        | 1      | yes         | no            | 
| 0              | Colombia       | Senegal        | 0      | no          | no            | 
| 1              | Japan          | Senegal        | 1      | yes         | no            | 
| 0              | Poland         | Colombia       | 0      | no          | no            | 
| 0              | Japan          | Colombia       | 0      | no          | no            | 
| 1              | Poland         | Japan          | 1      | no          | no            | 
| Total_Accuracy |                |                |        | 25/48       | 8/48          | 

The Group Table after all matches:

| Team           | Group | Point | Net Goals | Total Goals | 
|----------------|-------|-------|-----------|-------------| 
| Uruguay        | A     | 9     | 9         | 9           | 
| Russia         | A     | 4     | -2        | 2           | 
| Saudi Arabia   | A     | 4     | -2        | 2           | 
| Egypt          | A     | 0     | -5        | 0           | 
| Spain          | B     | 9     | 4         | 6           | 
| Morocco        | B     | 4     | 0         | 3           | 
| Portugal       | B     | 4     | 0         | 3           | 
| Iran           | B     | 0     | -4        | 0           | 
| Denmark        | C     | 7     | 3         | 5           | 
| France         | C     | 5     | 2         | 4           | 
| Australia      | C     | 4     | 0         | 3           | 
| Peru           | C     | 0     | -5        | 0           | 
| Argentina      | D     | 7     | 2         | 3           | 
| Croatia        | D     | 5     | 2         | 4           | 
| Iceland        | D     | 4     | 1         | 3           | 
| Nigeria        | D     | 0     | -5        | 0           | 
| Brazil         | E     | 9     | 4         | 4           | 
| Switzerland    | E     | 4     | 0         | 2           | 
| Serbia         | E     | 2     | -1        | 1           | 
| Costa Rica     | E     | 1     | -3        | 0           | 
| Germany        | F     | 9     | 5         | 6           | 
| Sweden         | F     | 6     | 0         | 3           | 
| Korea Republic | F     | 3     | -2        | 3           | 
| Mexico         | F     | 0     | -3        | 0           | 
| Belgium        | G     | 7     | 3         | 5           | 
| Tunisia        | G     | 6     | 2         | 5           | 
| England        | G     | 4     | 0         | 3           | 
| Panama         | G     | 0     | -5        | 0           | 
| Senegal        | H     | 9     | 3         | 4           | 
| Japan          | H     | 2     | -1        | 2           | 
| Poland         | H     | 2     | -1        | 1           | 
| Colombia       | H     | 2     | -1        | 0           | 


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
   
 
