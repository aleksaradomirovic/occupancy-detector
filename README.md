
# Milestone 2: Build First Agent

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
