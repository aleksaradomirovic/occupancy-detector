
# Milestone 2: Build First Agent

## Update: 
We added data preprocessing, model description, training, and CPTs sections based on gradescope feedback

### Data Preprocessing
Fortunately our dataset found on Kaggle is formatted well and does not require major preprocessing. However we did parse the csv file and store each column in its own Pandas Dataframe, as it is easy to perform operations on them this way. 

The dataset consists of 17,895 observations and 8 columns. The features include environmental metrics (Temperature, Humidity, Light, CO2, HumidityRatio), a date column, an id, and an Occupancy column. This dataset is well-suited for a classification problem, where the goal would be to predict Occupancy based on environmental factors. Since Light, CO2, and HumidityRatio are likely correlated with Occupancy, they may serve as strong predictive features.

### Model Description: 
Our model is structured as a **Naïve Bayes classifier**, where temperature serves as the primary feature for predicting room occupancy. The model assumes that the probability of occupancy given a temperature value follows the principles of **Bayes' Theorem**, meaning that we compute the likelihood $P(\text{Temperature} | \text{Occupancy})$, multiply it by the prior $P(\text{Occupancy})$, and normalize to obtain the posterior probability $P(\text{Occupancy} | \text{Temperature})$.

The correlations in the model are based on the observed relationship between temperature and occupancy in the training data. If higher temperatures are more likely when the room is occupied, the model captures this trend through the **likelihood function**. Since we grouped temperature into bins (`Temperature_group`), the model makes predictions by finding the closest temperature bin and comparing the probabilities of occupancy $P(1)$ and non-occupancy $P(0)$. This structure allows our model to generalize well by leveraging **conditional probabilities**, rather than making direct assumptions about the linearity of the relationship between temperature and occupancy.

### Training:
We trained our model using **Bayes’ Theorem** to predict room occupancy based on temperature. First, we discretized the `Temperature` variable into bins (`Temperature_group`) and computed the **likelihood** $ P(\text{Temperature} | \text{Occupancy}) $ by grouping the training data and normalizing the counts for each occupancy class. Next, we calculated the **prior probability** of occupancy $ P(\text{Occupancy}) $ by finding the proportion of occupied and unoccupied instances in the dataset. Using Bayes’ Theorem, we then derived the **posterior probability** $ P(\text{Occupancy} | \text{Temperature}) $ by multiplying the likelihood with the prior and normalizing the values so that probabilities sum to 1. Finally, we implemented a prediction function that takes a given temperature, finds the closest bin, retrieves the corresponding probabilities, and assigns the occupancy class with the highest probability. This approach allows for probabilistic predictions based on historical temperature distributions in the training data.


### CPTs: 
We calculated CPTs for all the columns in our dataset and the relationship they had conditionally to occupancy. These CPTs are included in our jupyter notebook linked [here](./Model1.ipynb). 






## Agent Definition
As smart building technology advances, real-time room occupancy detection is crucial for optimizing energy usage and space management. This project presents a probabilistic AI-driven approach to inferring room occupancy using environmental sensor data, including temperature, humidity, light levels, and CO₂ concentrations. Instead of relying on traditional machine learning classification models, we employ a Bayesian network to model the probabilistic dependencies between sensor readings and occupancy states. By structuring the problem within a probabilistic graphical model, we can dynamically update occupancy predictions as new data arrives, enabling robust inference under uncertainty.

Our approach leverages conditional probability distributions to incorporate prior knowledge about environmental variations and sensor reliability. Results from the Occupancy Detection dataset demonstrate the effectiveness of Bayesian inference in distinguishing occupied and unoccupied states with high accuracy. The proposed system could be integrated into Geisel’s existing room utilization display system, improving real-time space management. This research contributes to smart infrastructure by providing a scalable, uncertainty-aware solution for automated occupancy tracking, with potential applications in energy-efficient climate control and room allocation.


### Utility Based 
Agent makes decisions based on a utility function that evaluates different environmental sensor readings - such as temperature, humidity, and CO₂ levels - to determine the probability of a room being at maximum occupancy.


### Probabilistic Modeling
Our model is based on a Bayesian network, which utilizes conditional probability distributions to define the likelihood of different sensor values given occupancy states. The network dynamically updates its beliefs as new environmental data arrives, allowing it to make more informed occupancy predictions in real time.

### Conclusion Section
The model achieved an accuracy of **66.98%**, based **solely on temperature** indicating moderate predictive performance. However, the classification report suggests significant imbalances in precision and recall across classes, which affects overall reliability.

The model is effective at identifying unoccupied rooms, but its precision could be improved.
- 68% of predicted unoccupied rooms were actually unoccupied)
- 91% of actual unoccupied rooms were correctly identified
 
Model struggles significantly with detecting occupied rooms. It misses a large proportion of occupied cases, leading to false negatives.
- 26% of actual occupied rooms were correctly identified
- 61% of predicted occupied rooms were actually occupied

To improve performance over the next few iterations, we will analyze several different metrics concurrently (such as humidity, light levels, and CO₂. Additionally, using a more complex model could potentially provide more accurate predicitons.

[View Model 1 Notebook](./Model1.ipynb)
