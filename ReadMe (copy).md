# Pedestrian-Trajectory-Prediction-Papers

Collect SOTA Pedestrian Trajectory Prediction Papers, and some related papers in Behaviour Analysis. Continually but irregularly Updating, welcome to contact me if interested!



Sorted by field, then conferences/journals:

​	[Trajectory Prediction](#Trajectory-Prediction) 

​	[Behaviour Prediction](#Behaviour/Pose-Estimation)

***

## Trajectory Prediction

*CVPR 2023*: [[Conference]](https://ieeexplore.ieee.org/xpl/conhome/9811522/proceeding):

<br/>

*ICRA 2022*: [[Conference]](https://ieeexplore.ieee.org/xpl/conhome/9811522/proceeding):

<ul><details>  <summary>Meta-path Analysis on Spatio-Temporal Graphs for Pedestrian Trajectory Prediction <a href = "https://ieeexplore.ieee.org/document/9811632">[Paper]</a></summary>  <p>use Spatio-temporal graphs and <strong>meta-path</strong>(connecting nodes via temporal edges and spacial edges) to predict trajectory with BEV datasets(UCY/ETH)</p></details> </br><details>  <summary><strong>Crossmodal Transformer Based Generative Framework for Pedestrian Trajectory Prediction</strong> <a href = "https://ieeexplore.ieee.org/document/9812226">[Paper]</a></summary>  <p>Introduce <strong>Crossmodal Transformer</strong> as the encoder to handle multiple modalities(ego-vehicle motion, pedestrian trajectory, pose and intention), and a bezier curve based deocoder to prdict pedestrian trajectory. The whole archicture is CVAE, JAAD and PIE are used.</p></details>
</br><details>  <summary><strong>Grouptron: Dynamic Multi-Scale Graph Convolutional Networks for Group-Aware Dense Crowd Trajectory Forecasting</strong> <a href = "https://ieeexplore.ieee.org/document/9811585">[Paper]</a></summary>  <p>Use <strong>spatio-temporal graphs</strong>to creat 3 level of interactions of pedestrians(individual, intra-group, and inter-group), then use LSTM as individual encoder and <strong>STGCN</strong> as group encoder.</p></details></ul><br/>
*ICRL 2022:* [[Conference]](https://openreview.net/group?id=ICLR.cc/2022/Conference)

<ul><details>  <summary>Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories <a href = "https://openreview.net/pdf?id=Wm3EA5OlHsG">[Paper]</a></summary>  <p>The paper introduces <strong>joint</strong> trajectory prediction for vehicles using Transformer</p></details> </br><details>  <summary>THOMAS: Trajectory Heatmap Output with learned Multi-Agent Sampling <a href = "https://arxiv.org/abs/2110.06607">[Paper]</a></summary>  <p>Propose a hierarchical heatmap decoder for vehicle trajectory prediction, allowing for unconstrained heatmap generation with optimized computational costs, enabling efficient simultaneous multi-agent prediction.</p></details></ul><br/>

*CVPR 2022:* [[Conference]](https://cvpr2022.thecvf.com/)

<ul><details>  <summary><strong>Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion </strong><a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Gu_Stochastic_Trajectory_Prediction_via_Motion_Indeterminacy_Diffusion_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>Stochastic trajectory predicition models the inferred uncertainty of pedestrians’ movements in every time frame. The paper presents a new stochastic trajectory prediction framework with <strong>motion indeterminacy diffusion</strong>.A Transformer-based decoder is used and encoder is employed from Trajectorn++</p></details> </br>
<details>  <summary><strong>Non-Probability Sampling Network for Stochastic Human Trajectory Prediction</strong><a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Bae_Non-Probability_Sampling_Network_for_Stochastic_Human_Trajectory_Prediction_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>Propose a novel sampling methods which adopts Quasi-Monte Carlo (QMC) sampling to generate a set of randon latent vectors.The proposed Non-Probability Sampling Network (NPSN) works as purposive sampling, which relies on the past trajectories of pedestrians when selecting samples in the distribution. GAT is used to aggregate the features for neighbors by assigning different importance to their
edge(with attention). </p></details></br>
<details>  <summary>Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Towards_Robust_and_Adaptive_Motion_Forecasting_A_Causal_Representation_Perspective_CVPR_2022_paper.pdf">[Paper]</a></summary>  </details></br>
<details>  <summary><strong>GroupNet: Multiscale Hypergraph Neural Networks for
Trajectory Prediction with Relational Reasoning</strong> <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Bae_Non-Probability_Sampling_Network_for_Stochastic_Human_Trajectory_Prediction_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>GroupNet uses attention mechanism to infer multiple <strong>hypergraphs</strong> according to group size, and introduce interaction embedding using neural interaction strength, category and its function, which are updated during iterations. NBA SportVU, ETH/UCY and SSD are used.</p></details></br>
<details>  <summary>How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting
 <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Monti_How_Many_Observations_Are_Enough_Knowledge_Distillation_for_Trajectory_Forecasting_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>The paper proposed to use knowledge distillation to reduce input length while improving accuracy for trajectory prediction.The motivation is input errors from tracking and detection(using machine perception) accumulate during inference time(even training set is manually corrected). <strong>Spacial-temporal Transformer</strong> is employed as the vanilla model, and KD is applied on it. Datasets: ETH/UCY, SSD, and Lyft Prediction Dataset</p></details></br>
<details>  <summary>Human Trajectory Prediction with Momentary Observation
 <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Human_Trajectory_Prediction_With_Momentary_Observation_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>The authors consider an extreme case that only two frames are available for prediction, named Momentary Trajectory Prediction. The model carefully models local&amp;global context and velocity of pedestrians into semantic map, then encode them with Transformer, before using these feature for <strong>context restoration</strong> and trajectory prediction(multitask). The context restoration task encourage the model to perceive comprehensive social and scene context information.Datasets: ETH/UCY and SSD.</p></details></br>
<details>  <summary>Remember Intentions: Retrospective-Memory-based Trajectory Prediction
 <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Remember_Intentions_Retrospective-Memory-Based_Trajectory_Prediction_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>MemoNet designsa two-step trajectory prediction system, where the first step is to leverage MemoNet to predict the destination and the second step is to fulfill the whole trajectory according to the predicted destinations. a pair of memory banks to explicitly store representative instances in the training set, acting as prefrontal cortex in the neural system, and a trainable memory addresser to adaptively search a current situation with similar instances in the memory bank, acting like basal ganglia. The two-step trajectory prediction system is proposed, where the first step is to leverage MemoNet to predict the destination and the second step is to fulfill the whole trajectory according to the predicted destinations. Datasets: ETH/UCY, SSD, and NBA.</p></details></br>
<details>  <summary>Adaptive Trajectory Prediction via Transferable GNN
 <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Adaptive_Trajectory_Prediction_via_Transferable_GNN_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>The paper lies in trajectory <strong>domain shift</strong>m and proposes a <strong>transferable graph neural network(GNN)</strong> via adaptive knowledge learning, which is based on attention and multiple loss calculation. Datasets: ETH/UCY</p></details></br>
<details>  <summary>Importance is in your attention:
agent importance prediction for autonomous driving
 <a href = "https://openaccess.thecvf.com/content/CVPR2022W/Precognition/papers/Hazard_Importance_Is_in_Your_Attention_Agent_Importance_Prediction_for_Autonomous_CVPRW_2022_paper.pdf">[Paper]</a></summary>  <p>The paper shows that the attention information can also be used to measure the importance of each agent with respect to the ego vehicle’s future planned trajectory.Datasets: NuPlans</p></details></br>
<details>  <summary><strong>Graph-based Spatial Transformer with Memory Replay for Multi-future
Pedestrian Trajectory Prediction</strong> <a href = "https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Graph-Based_Spatial_Transformer_With_Memory_Replay_for_Multi-Future_Pedestrian_Trajectory_CVPR_2022_paper.pdf">[Paper]</a></summary>  <p>Design two Transformers(temporal and spacial, in spaicial Transformer they used graph representations and <strong>Transformer-graph convolution</strong>) to handle temporal and spacial features respectively, then stack them to model the spacial-temporal interaction of pedestrians. A simple decoder is used and external graph memory(hold graph representations after encoding and it is added to temporal Transformer) is introduced to handle long-term temporal consistency. </p></details></br>
</ul>

*ECCV(European Conference on Computer Vision) 2020:* [[Conference]](https://www.ecva.net/papers.php)

<ul>
<details>  <summary><strong>Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction</strong> <a href = "https://arxiv.org/pdf/2005.08514.pdf">[Paper]</a></summary>  <p>Design two Transformers(temporal and spacial, in spaicial Transformer they used graph representations and <strong>Transformer-graph convolution</strong>) to handle temporal and spacial features respectively, then stack them to model the spacial-temporal interaction of pedestrians. A simple decoder is used and external graph memory(hold graph representations after encoding and it is added to temporal Transformer) is introduced to handle long-term temporal consistency. </p></details></br></ul>
*ECCV(European Conference on Computer Vision) 2022:* [[Conference]](https://www.ecva.net/papers.php)

<ul><details>  <summary><strong>Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction </strong><a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820263.pdf">[Paper]</a></summary>  <p>The paper models <strong>pedestrian interaction hierarchically</strong> into 3 level, and introduce a group pooling and unpooling scheme that can be used on baseline models(GCN,GAT,ST-Transformer), presenting SOTA results. More details can be found in the independent note.</p></details> </br>
<details>  <summary>S2F2: Single-Stage Flow Forecasting for Future Multiple Trajectories Prediction <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820593.pdf">[Paper]</a></summary>  <p>The paper consider trajectory prediction as a 2-step task, where the first is detection and association(tracking) and the second step is prediction, so they propose a framework that detects and predicts pedestrians' trajectory simultaneously.They select a subset of <strong>MOT17 and MOT20</strong>, where the image sequences are sourced from static cameras.</p></details></br>
<details>  <summary>Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820125.pdf">[Paper]</a></summary><p>
    The paper proposes to use <strong>hierarchical latent space</strong> to mitigate the <strong>blurry sample generation problem</strong> in <strong>VAE-based</strong> model for trajectory prediction(for vehicles). The low-level latent variable is employed to model each mode of the mixture and the high-level latent variable is employed to represent the weights for the modes. To model each mode accurately, they condition the low-level latent variable using two lane-level context vectors (one corresponds to vehicle-lane interaction (VLI) and the other to vehicle-vehicle interaction (V2I)) computed in novel ways.</p> </details></br>
<details>  <summary><strong>Action-based Contrastive Learning for Trajectory Prediction</strong> <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990140.pdf">[Paper]</a></summary>  <p>The paper introduces a novel Action-based Contrastive Loss as an additional regularisation in <strong>CVAE-based models</strong>s for <strong>ego-centric view</strong> pedestrian prediction. The fundamental idea behind this new loss is that trajectories of pedestrians performing the same action should be closer to each other in the feature space than the trajectories of pedestrians with significantly different actions. <strong>TITAN,PIE and JAAD</strong> are used.</p></details></br>
<details>  <summary>Entry-Flipped Transformer for Inference and Prediction of Participant Behavior
 <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640433.pdf">[Paper]</a></summary>  <p>The paper proposes a novel Entry-Flipped Transformer (EF-Transformer), which tackle the problem of error accumulation by flipping the order of query, key, and value entries, to increase the importance and fidelity of observed features in the current frame. The applied domain is estimating how a set target participants react to the behaviour of other observed participants, like two-person dance, sports(tennis), and related datasets(and ETH) are used.</p></details></br>
<details>  <summary>(Metric)Social-Implicit: Rethinking Trajectory Prediction Evaluation and The Effectiveness of Implicit Maximum Likelihood Estimation
 <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820451.pdff">[Paper]</a></summary> </details></br>
<details>  <summary>Social-SSL: Self-Supervised Cross-Sequence Representation Learning Based on Transformers for Multi-Agent Trajectory Prediction
 <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820227.pdf">[Paper]</a></summary>  <p>Most <strong>Transformer-based</strong> networks are trained end-to-end without capitalizing on the value of pre-training, but this paper proposes to Social-SSL that captures <strong>cross-sequence trajectory structures</strong> via self-supervised <strong>pre-training</strong>.The model includes 3 pre-text task: interaction type prediction and closeness prediction are designed to capture the inter-relation between the target agent and each of the social agents. Meanwhile, masked cross-sequence to sequence pre-training provides the understanding of intra-relation among the remaining sequence of the target agent. By sharing the representations/features, the model  is effective for crowded agent scenarios and can reduce the amount of data needed for fine-tuning. </p></details></br>
<details>  <summary>(vehicle)Social ODE: Multi-Agent Trajectory Forecasting with Neural Ordinary Differential Equations
 <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820211.pdf">[Paper]</a></summary> </details></br>
<details>  <summary><strong>View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums</strong>
 <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820661.pdf">[Paper]</a></summary>  <p>This work studies agents’ trajectories in a “vertical” view, i.e., modeling and forecasting trajectories from the <strong>spectral domain</strong>. V2-Net contains two sub-networks, to hierarchically model and
predict agents’ trajectories with trajectory spectrums. The coarse-level
keypoints estimation sub-network first predicts the “minimal” spectrums
of agents’ trajectories on several “key” frequency portions. Then the finelevel spectrum interpolation sub-network interpolates the spectrums to reconstruct the final predictions</p></details></br>
<details>  <summary><strong>SocialVAE: Human Trajectory Prediction using Timewise Latents</strong> <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640504.pdf">[Paper]</a></summary>  <p>The core of SocialVAE is a <strong>RNN-based</strong> timewise variational autoencoder architecture that exploits stochastic recurrent neural networks to perform prediction, combined with a social attention mechanism and a backward posterior approximation to allow for
better extraction of pedestrian navigation strategies. The model uses latent variables as
stochastic parameters to condition the hidden dynamics of RNNs at each time
step, in contrast to previous solutions that condition the prior of latent variables
only based on historical observations. </p></details></br>
<details>  <summary>(vehicle)D2-TPred: Discontinuous Dependency for
Trajectory Prediction under Traffic Lights <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680512.pdf">[Paper]</a></summary>  <p>Vehicles do not exhibit continuety of movement under the traffic lights(discontinuous dependency). The paper proposes a vehicle trajectory prediction approach with respect to traffic lights, which uses a spatial dynamic interaction graph (SDG) as well as a behavior dependency graph (BDG) to handle the problem of discontinuous dependency in the spatial-temporal space. The paper also provides a novel dataset for vehicle trajectory prediction at intersections. Graphs are handled by <strong>GAT</strong> </p></details></br>
<details>  <summary>Human Trajectory Prediction via Neural Social
Physics
<a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940368.pdf">[Paper]</a></summary>  <p>Tha paper proposes a new method combining both <strong>rule-based</strong> and learning-based methods based on a new Neural Differential Equation model.The whole contains pre-trained networks for goal sampling,collision avoidance, and context extraction, in which features are fed to the proposed NSP. Combining with a CVAE, the model output trajectory distribution. Dataset:UCY/ETH, SSD. </p></details></br>
<details>  <summary>(vehicle)Aware of the History: Trajectory Forecasting
with the Local Behavior Data <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820383.pdf">[Paper]</a></summary>  <p></p></details></br>
</ul>

*IEEE Transactions on Intelligent Transportation Systems(2020-now):*  [[Journal]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979)

<ul><details>  <summary>Pedestrian Models for Autonomous Driving Part II: High-Level Models of Human Behavior<a href = "https://ieeexplore.ieee.org/document/9151337">[Paper]</a></summary></details> </br>
<details>  <summary>Review of Pedestrian Trajectory Prediction Methods: Comparing Deep Learning and Knowledge-Based Approaches <a href = "https://ieeexplore.ieee.org/document/9899358">[Paper]</a></summary> <p></p></details></br>
<details><summary><strong>BR-GAN: A Pedestrian Trajectory Prediction Model Combined With Behavior Recognition</strong><a href = "https://ieeexplore.ieee.org/document/9851641">[Paper]</a></summary><p>This paper builds the environmental, the social and the behaviour feature modules based on the <strong>GAN</strong> framework to process information, where the<strong>behaviour recognition module</strong> is a YOLOV3 pre-trianed with a human behaviour dataset.
</p></details> </br>
<details><summary>Pedestrian Motion Trajectory Prediction in Intelligent Driving from Far Shot First-Person Perspective Video<a href = "https://ieeexplore.ieee.org/document/9340008">[Paper]</a></summary><p>This paper introduces a task about trajectory prediction from <strong>far shot first-person perspective</strong> video. The FPL model contains: macroscopic pedestrian trajectory prediction module under the close correlation between neighboring frames(CNN), A relative motion transformation module(ego-vehicle motion), and a new far shot first-person pedestrian motion dataset.
</p></details> </br>
<details><summary>Holistic LSTM for Pedestrian Trajectory Prediction<a href = "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9361440">[Paper]</a></summary><p>The paper proposes a novel structure of LSTM, named Holistic LSTM, where memory cells model the inter-related dynamics of pedestrian intention, vehicle speed and global scene dynamics among the temporal frame
sequences. The whole model includes intention module(ConvLSTM and LSTM), correlation module(optical flow), and ego-motion module.
</p></details> </br>
<details><summary><strong>Pedestrian Graph +: A Fast Pedestrian Crossing Prediction Model Based on Graph Convolutional Networks</strong><a href = "https://ieeexplore.ieee.org/document/9774877">[Paper]</a></summary><p>Pedestrian Graph +, a <strong>multimodal GCN-based</strong> model that processes the human pose estimation data in real-time (taking into account the non-euclidean geometry). To alleviate the context loss in GCN, cropped images, cropped segmentation maps, egovehicle velocity data are added to the model. The main branch uses <strong>2D/3D human key points</strong> graph(obtained by <strong>AlphaPose</strong>) as the input, and the output are the following 30 poses, which are used to predict crossing/not crossing event.
problem)
</p></details> </br>
<details><summary>Causal Temporal–Spatial Pedestrian Trajectory Prediction With Goal Point Estimation and Contextual Interaction<a href = "https://ieeexplore.ieee.org/document/9896809">[Paper]</a></summary><p>The network comprises a spatio-temporal Transformer (baseline) and Visual Intention Knowledge Refinement to comprehend human intention and formulates decision-making processes. Visual Intention Knowledge is constructed through a Collision Perception (CP) module and a Swerve Optimization (SO) module, which receive the output from Transformer as pedestrian direction estimation and bounding boxes, then predict intention field and intention refinement respectively.
    </p></details> </br>
<details><summary>Trajectory Forecasting Based on Prior-Aware Directed Graph Convolutional Neural Network<a href = "https://ieeexplore.ieee.org/document/9686621">[Paper]</a></summary><p>The paper presents a directed(assuming the strength of influence between agents are not equal, like cars and pedestrians) graph convolutional neural network for multiple agents trajectory prediction. The three directed graph topologies are view graph, direction graph, and rate graph, by encoding different prior knowledge of a cooperative scenario, which endows the capability of our framework to effectively characterize the asymmetric influence between agents. 
</p></details> </br>
<details><summary>Trajectory Prediction for Autonomous Driving Using Spatial-Temporal Graph Attention Transformer<a href = "https://ieeexplore.ieee.org/document/9768029">[Paper]</a></summary><p>
</p></details> </br>
<details><summary>Long-Short Term Spatio-Temporal Aggregation
for Trajectory Prediction<a href = "https://ieeexplore.ieee.org/document/10018105">[Paper]</a></summary><p> The paper first introduce a GNN-based spacial encoder, and then proposes LSSTA utilizing a transformer network to handle long-term temporal dependencies
and aggregates the spatial and temporal features with a temporal
convolution network (TCN). Scene information and future trajectory are added when training.
</p></details> </br>
<details><summary>Human Trajectory Forecasting in Crowds: A Deep Learning Perspective<a href = "https://ieeexplore.ieee.org/document/9408398">[Paper]</a></summary><p>
</p></details> </br>
<details><summary>Fully Convolutional Encoder-Decoder With an Attention Mechanism for Practical Pedestrian Trajectory Prediction<a href = "https://ieeexplore.ieee.org/document/9768201">[Paper]</a></summary><p>
</p></details> </br>
</ul>

*IEEE Transactions on Intelligent Vehicles(2020-now)*:  [[Journal]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7274857)

<ul>
    <details><summary>Pedestrian Trajectory Prediction Combining Probabilistic Reasoning and Sequence Learning<a href = "https://ieeexplore.ieee.org/document/8957246">[Paper]</a></summary><p>This early paper present a framework combining <strong>Dynamic Bayesian Network and Seq2Seq</strong> model to predict pedestrian crossing at non-signalised intersections. The causal relationships among the environmental clues and the pedestrian’s motion decision are represented with DBN graph. The LSTM-based Seq2Seq model is used to predict trajectory.
</p></details> </br>
</ul>


## Behaviour/Pose Estimation

*CVPR 2023：*

<ul>
<details>  <summary><strong>Action-based Contrastive Learning for Trajectory Prediction</strong> <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990140.pdf">[Paper]</a></summary>  <p>The paper introduces a novel Action-based Contrastive Loss as an additional regularisation in <strong>CVAE-based models</strong>s for <strong>ego-centric view</strong> pedestrian prediction. The fundamental idea behind this new loss is that trajectories of pedestrians performing the same action should be closer to each other in the feature space than the trajectories of pedestrians with significantly different actions. <strong>TITAN,PIE and JAAD</strong> are used.</p></details></br></ul>

*ECCV 2022:* [[Conference]](https://www.ecva.net/papers.php)

<ul><details>  <summary><strong>Action-based Contrastive Learning for Trajectory Prediction</strong> <a href = "https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990140.pdf">[Paper]</a></summary>  <p>The paper introduces a novel Action-based Contrastive Loss as an additional regularisation in <strong>CVAE-based models</strong>s for <strong>ego-centric view</strong> pedestrian prediction. The fundamental idea behind this new loss is that trajectories of pedestrians performing the same action should be closer to each other in the feature space than the trajectories of pedestrians with significantly different actions. <strong>TITAN,PIE and JAAD</strong> are used.</p></details></br>
</ul>

*IEEE Transactions on Intelligent Transportation Systems(2020-now):*  [[Journal]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979)

<ul><details><summary><strong>BR-GAN: A Pedestrian Trajectory Prediction Model Combined With Behavior Recognition</strong><a href = "https://ieeexplore.ieee.org/document/9851641">[Paper]</a></summary><p>This paper builds the environmental, the social and the behaviour feature modules based on the <strong>GAN</strong> framework to process information, where the<strong>behaviour recognition module</strong> is a YOLOV3 pre-trianed with a human behaviour dataset.
</p></details> </br>
<details><summary>Pedestrian Path, Pose, and Intention Prediction Through Gaussian Process Dynamical Models and Pedestrian Activity Recognition<a href = "https://ieeexplore.ieee.org/document/8370119">[Paper]</a></summary><p>
</p></details> </br></ul>


*CoRL (Conference on Robot Learning) 2022:* [[Conference]](https://proceedings.mlr.press/v205/)

<ul><details>  <summary><strong>HUM3DIL: Semi-supervised Multi-modal 3D Human
Pose Estimation for Autonomous Driving</strong> <a href = "https://arxiv.org/pdf/2212.07729.pdf">[Paper]</a></summary>  <p>See details in the independent note.</p></details></br>
<details>  <summary>（Decoder)Towards Capturing the Temporal Dynamics for
Trajectory Prediction: a Coarse-to-Fine Approach
<a href = "https://openreview.net/pdf?id=PZiKO7mjC43">[Paper]</a></summary>  <p> The paper pints out SOTA works usually use MLP decoder to output a TimeX2 Tensor, which ignored the temporal correlation among the future time-steps. However, they found autoregressive RNN decoder leads to a performance drop even using teaching forcing or history highway, but when first using an MLP to generate a scratch trajectory and then using a structure with temporal inductive bias (RNN/1D-CNN) to refine it, the SOTA model’s performance could be boosted. Anuthors also examine several objective functions to add temporal priors to the output. We report the notable improvements by first generating velocity and then accumulating temporally into coordinates to calculate per-step loss.</p></details></br>
</ul>

*IROS(Conference on Intelligent Robots and Systems) 2022:* [[Conference]](https://ieeexplore.ieee.org/xpl/conhome/9981026/proceeding)

<ul><details>  <summary><strong>Pedestrian Intention Prediction Based on Traffic-Aware Scene Graph Model</strong> <a href = "https://ieeexplore.ieee.org/abstract/document/9981690">[Paper]</a></summary>  <p>The paper introduces to construct scene graph to model pedestrian-pedestrian interactions and pedestrian-traffic state interactions(in this paper, traffic lights). The rest of model contains GRU and attention mechannism to extract temporal representations. </p></details></br>
</ul>

*ACCV(Asian Conference on Computer Vision) 2022:* [[Conference]]([[Conference]](https://ieeexplore.ieee.org/xpl/conhome/9981026/proceeding))

<ul><details>  <summary><strong>Social Aware Multi-Modal Pedestrian Crossing Behavior Prediction</strong> <a href = "https://openaccess.thecvf.com/content/ACCV2022/papers/Zhai_Social_Aware_Multi-Modal_Pedestrian_Crossing_Behavior_Prediction_ACCV_2022_paper.pdf">[Paper]</a></summary>  <p>The paper proposes Multi-Modal Conditional Generative Module to learn multiple modes of pedestrian future actions, and <strong>a spatial-temporal heterogeneous graph(that is social-aware)</strong> to model spatial-temporal interactions between the target pedestrian and heterogeneous traffic objects.</p></details></br>
</ul>

*WACV 2021*: [[Conference]](https://wacv2021.thecvf.co

<ul><details>  <summary><strong>Benchmark for Evaluating Pedestrian Action Prediction</strong> <a href = "https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf">[Paper]</a></summary>  <p>A benchmark of <strong>pedestrian action prediction(Crossing or not)</strong> for PIE and JAAD.  </p></details></br>
</ul>

*IEEE Transactions on Human-Machine Systems*: [[Journal]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221037)

<ul><details>  <summary>Intent Prediction in Human–Human Interactions <a href = "https://ieeexplore.ieee.org/document/10037766">[Paper]</a></summary>  <p> </p></details></br>
</ul>

*ICRA 2022*: [[Conference]](https://ieeexplore.ieee.org/xpl/conhome/9811522/proceeding)

<ul><details>  <summary>Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion<a href = "https://ieeexplore.ieee.org/document/9811664">[Paper]</a></summary>  <p> </p></details></br>
</ul>

*ICRA 2021:*

<ul><details>  <summary><strong>Graph-SIM: A Graph-based Spatiotemporal Interaction Modelling for Pedestrian Action Prediction </strong><a href = "https://arxiv.org/abs/2012.02148">[Paper]</a></summary>  <p> </p></details></br>
</ul>

*IEEE Transactions on Intelligent Transportation Systems(2020-now):*  [[Journal]](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979)

<ul><details>  <summary><strong>Crossing or Not? Context-Based Recognition of Pedestrian Crossing Intention in the Urban Environment</strong><a href = "https://ieeexplore.ieee.org/document/9345505">[Paper]</a></summary>  <p> </p></details></br>
<details>  <summary>Spatiotemporal Relationship Reasoning for
Pedestrian Intent Prediction
<a href = "https://arxiv.org/pdf/2002.08945.pdf">[Paper]</a></summary>  <p> </p></details></br>
<details>  <summary>Pedestrian Crossing Intention Prediction at Red-Light Using Pose Estimation<a href = "https://ieeexplore.ieee.org/document/9423518">[Paper]</a></summary>  <p> </p></details></br>
</ul>

*Arxiv:*

<ul><details>  <summary><strong>PedFormer: Pedestrian Behaviour Prediction via Cross-Modal Attention Modulation and Gated Multitask Learning</strong><a href = "https://arxiv.org/pdf/2210.07886.pdf">[Paper]</a></summary>  <p> </p></details></br>
</ul>
