# Shape-Controlled Clustering and Christofides algorithm for Capacited Vehicle Routing Problem
 
This repository is derived from the winning code of the A2A Challenge 2020 organized at the University of Milano-Bicocca.
It was realized in collaboration with Matteo De Giosa and Andrea Santoro.

The goal of the challenge was to implement a solution to a Capacited Vehicle Routing Problem with the following constraints:
- Each bin need to be emptied at least once a day
- Minimize the emptyings of every bins
- Minimize the travel distance of every vehicle

We proposed a solution based on a modified k-Means clustering to group the bins to be emptied by a vehicle and the Christofides algorithm to find the optimal path to empty them.

For example, in the following figure there are the normal k-Means clustering on the left and the modified k-Means on the right. The second one is a lot better as total distance travelled by the vehicles.

![Routing examples](https://github.com/gianlucapagliara/SC3-CVRP/blob/main/output/Routing%20examples.jpg)

