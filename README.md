# Applied AI for Smart City Solutions

This repository contains an applied Artificial Intelligence project focused on **smart city optimisation**, combining **intelligent traffic management** and **energy consumption prediction** using a variety of AI techniques.

The project is structured around four core components (Parts A–D), each addressing a specific problem statement and learning outcome using practical implementations in Python.

---

##  Project Overview

### Problem Statement 1  
**Intelligent Traffic Management for Smart Cities**

Autonomous vehicles navigate a city road network represented as a graph. Each route decision considers multiple weighted factors such as:
- Distance
- Traffic congestion
- Accident likelihood
- Road safety

### Problem Statement 2  
**Predicting Energy Consumption in Smart Buildings**

Historical household electricity usage data is used to model and predict future energy consumption using both **deep learning** and **decision-theoretic approaches**.

---

## 📁 Repository Structure

```text
applied-ai-smart-city-solutions/
├─ PartA/
│  └─ hill_climbing.py
├─ PartB/
│  ├─ aco.py
│  └─ graph_utils.py
├─ PartC/
│  ├─ evaluation.py
│  ├─ lstm_forecast.py
│  ├─ preprocess_and_save.py
│  └─ train_and_tune.py
├─ PartD/
│  └─ mdp_energy_modeling.py
├─ path_visualizations/
│  └─ (example result images)
├─ results/
│  ├─ aco_vs_worst_results.csv
│  └─ hillclimbing_vs_randomwalk_results.csv
├─ README.md
├─ requirements.txt
└─ .gitignore


```

### Additional folders and files include:

* path_visualizations/: Generated plots comparing routing strategies

* .pth, .npy, .csv files: Saved models, predictions, and experimental results

* requirements.txt: Python dependencies

## Part A: Search Algorithm for Path Finding (Hill Climbing)

* Folder: PartA/

### Requirements:

* Model a road network as a graph with distance, congestion, accident likelihood, and safety costs

* Implement Hill Climbing to find an optimal path

* Apply random restart to avoid local optima

* Output the final route and total cost

### Implementation:

* "hill_climbing.py"
Implements the Hill Climbing algorithm with a weighted cost function and random restarts.

* Experimental results are compared against a random walk baseline.

* Visual route comparisons are available in path_visualizations/.


## Part B: Swarm Intelligence for Traffic Optimization (ACO)

* Folder: PartB/

### Requirements:

* Select and justify a swarm intelligence algorithm

* Define swarm representation and navigation

* Implement update rules

* Specify convergence criteria

* Analyse performance against an alternative method


### Implementation:

* Algorithm Used: Ant Colony Optimization (ACO) 

* "aco.py" Implements pheromone-based route optimization considering congestion and safety.

* "graph_utils.py" 
Shared graph utilities for route evaluation.

* Performance comparisons are stored in "aco_vs_worst_results.csv".


## Part C: Recurrent Neural Networks for Time-Series Forecasting

* Folder: PartC/

### Requirements:

* Preprocess and normalize time-series data

* Design and train an RNN (LSTM/GRU)

* Predict future energy consumption

* Perform hyperparameter tuning

* Compare against a moving average baseline


### Implementation:

* Dataset: Household Electric Power Consumption ([UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption))

* "preprocess_and_save.py"
Data cleaning, normalisation, and sequence generation.

* "lstm_forecast.py"
LSTM model architecture definition.

* "train_and_tune.py"
Model training and hyperparameter experimentation.

* "evaluation.py"
Model evaluation using MSE and baseline comparison.

* Trained models and predictions are saved as ".pth" and ".npy" files.




## Part D: Markov Decision Process for Energy Consumption Modeling

* Folder: PartD/

### Requirements:

* Define energy consumption states

* Construct transition probability matrices

* Define a reward function

* Apply policy or value iteration

* Compare results with another forecasting method


### Implementation:


* "mdp_energy_modeling.py"
Implements:
 1. State discretisation from historical energy data
 2. Transition probability estimation
 
 3. Reward-based optimisation
 
 4. Value Iteration to derive an optimal policy



## Technologies Used

* Python 3.10

* NumPy, Pandas

* PyTorch

* Matplotlib

* Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```


## Results & Visualisations

* Route optimisation comparisons (Hill Climbing vs Random Walk)

* Swarm intelligence performance metrics

* Energy consumption predictions and true vs predicted plots

* Stored outputs in .csv, .png, .npy, and .pth formats


## Future Extensions

* Integrate real-time traffic data
* Extend ACO with dynamic pheromone evaporation
* Replace LSTM with Transformer-based forecasting
* Combine MDP with reinforcement learning (Q-learning / DQN)

## License 
This project is released for educational and portfolio purposes.