## Simple k-nearest neighbour classifier implementation with sklearn

In this project, I am trying to explain KNN in a simplified manner.

#### Problem: 

	A person of age(x) and height(h) drink milk?

#### Expected Answer:
	
	Yes (or) No

#### Training Data: [ data/training_age_vs_like_milk.csv  ]
	
	The training data contains a list of person's age, height, and their corresponding liking towards milk. Here we take age and height as input and their preference towards milk as output.

#### Testing Data: [ data/test_age_vs_like_milk.csv ]

	The testing data contains a list of person's age, height, and their corresponding liking towards milk. Here we take age and height as input and their preference towards milk as output.
	
#### Source Code:

	- src/knn.p

## How to run this program?

#### Setup local python environment: [ Tested only on Mac ]

    1. cd <this repo>
    2. cd src
    3. chmod +x setup_mac.sh 
    4. ./setup_mac.sh


#### Run KNN program: [ Tested only on Mac ]
    1. cd <this repo>
    2. cd src
    3. chmod +x run.sh 
    4. ./run.sh

#### Clean local python environment: [ Tested only on Mac ]
    1. cd <this repo>
    2. cd src
    3. chmod +x clean.sh
    4. ./clean.sh


