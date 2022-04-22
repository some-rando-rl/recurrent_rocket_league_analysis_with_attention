# Next goal predictor
## Model
 - attention?
### Observation space
 - basically a game_state
 - calculate has_flip?
 - explain parts of game state
## Data
 - have data split into ranks(GC, SSL, RLCS)
 - Structure:
   - Rank -> Replay -> Chronological Goal Segments -> Chronological Game States
 - have augmentation. that is Orange = Blue, mirroring
#Hyper-parameters
 - search for learning rate
 - as big of batch size as possible
 - ADAM vs SGD

## Logging
 - Track loss
 - Track update magnitude
 - Track different runs and their hyper-parameters
 
## What do I want to be able to do
 - I want to see accuracy of model in different times before goal
 - Calculate significance of touches of a player(Difference of P(next_goal) according to the model )
 - Distribution of classes(Who scored) in a given replay group
 - Generate heatmap of the positions with maximum P(next_state)

## Implement Captum or other model comprehension tool