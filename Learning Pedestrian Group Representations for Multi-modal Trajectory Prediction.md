**Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction**: 

​	Representation available: [here](https://inhwanbae.github.io/publication/gpgraph/)

​	Domain: Group behaviour in pedestrian trajectory prediction

​	Motivation/Problem: Pioneering learning-based models share the group features at individual nodes in graphs(purple cross edges in the figure). This makes models hard to represent group features because of the overwhelming number of edges, especially in crowded scenarios.



![Fig. 1.](https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-3-031-20047-2_16/MediaObjects/539987_1_En_16_Fig1_HTML.png)

​	Method: 

 1. Group Assignment: measure the similarity among pedestrian pairs(L2 distance), and define a learnable thresholding parameter π to determine how close pedestrians will be considered as a group. For each group, the most representative  feature each pedestrian is selected via an **average pooling** then aggregated as the group trajectory feature. To assign features to each group member(unpooling), duplicate the group features and then assign them into nodes for all the relevant group members so that they have identical group behaviour information.

    ![Image](https://inhwanbae.github.io/assets/img/gpgraph/gpgraph-poolingunpooling.svg)

 2. Straight-Through Group Estimator: 

​		Current problem:  index information is not treated as learnable parameters in Group Assignment.

​		Inspiration: biased path derivative estimator(see in [this](https://arxiv.org/pdf/1308.3432.pdf))

​		Process: In forward, group pooling over both pedestrian features and group index. In backward, for each pair in distance matrix D, compute the probability that a pair of pedestrians belongs to the same group using the proposed differentiable binary thresholding function 1/(1+exp-(π-x)), (a variant sigmoid?) then measure the normalized probability A of the summation of all neighbours’ probability. Then compute a new pedestrian trajectory feature X′ by aggregating features between group members through the matrix multiplication of X and A.

![Image](https://inhwanbae.github.io/assets/img/gpgraph/gpgraph-straightthrough.svg)
$$
\Large{\boldsymbol {A}}_{\, i,j} = \frac {\frac {1} {1+{e}^{(\frac {{D}_{i,j}-\pi } {\tau })}}} {\sum ^{N}_{i=1} ({\frac {1} {1+{e}^{(\, \frac {{D}_{i,j}-\pi } {\tau })}}})}
$$

  3. Hierarchical Graphs: 

     ![Image](https://inhwanbae.github.io/assets/img/gpgraph/gpgraph-hierarchy.svg)