# Shape-Controlled Clustering and Christofides algorithm for Capacited Vehicle Routing Problem
 
This repositoty is derived from the winning code of the A2A Challenge 2020 organized at the University of Milano-Bicocca.
It was realized in collaboration with Matteo De Giosa and Andrea Santoro.

The goal of the challenge was to implement a solution to a Capacited Vehicle Routing Problem with the following constraints:
- Each bin need to be emptied at least once a day
- Minimize the emptyings of every bins
- Minimize the travel distance of every vehicle

We proposed a solution based on a modified K-Means clustering to group the bins to be emptied by a vehicle and the Christofides algorithm to find the optimal path to empty them.
