# Phycics-informed-NN-for-Power-Systems

We introduce  a  framework  for  physics-informed  neural  networks in power system applications. Exploiting the underlying physical laws  governing  power  systems,  and  inspired  by  recent  developments  in  the  field  of  machine  learning, we propose  a neural network training procedure that can make use of the wide range of mathematical models describing power system behavior, both  in  steady-state  and  in  dynamics.  Physics-informed  neural networks  require  substantially  less  training  data  and  result  in much  simpler  neural  network  structures,  while  achieving  high accuracy.  This  work  unlocks  a  range  of  opportunities  in  power systems,  being  able  to  determine  dynamic  states,  such  as  rotor angles and frequency, and uncertain parameters such as inertia and  damping  at  a  fraction  of  the  computational  time  required by conventional methods. We focus on introducing the framework  and  showcases  its  potential  using  a  single-machine infinite bus system as a guiding example. Physics-informed neural networks  are  shown  to  accurately  determine  rotor  angle  and frequency  up  to 87 times faster than  conventional  methods.


The folder `continuous_time_inference’ corresponds to the results presented in Section III.B. First, we load the input data file (`swingEquation_inference.mat’). Then, we randomly define the training set based on the number of Nu.  After the training process, the variables U_pred and Exact contain the predicted and actual values of the angle trajectories, respectively. The code also provides the L2 error between exact and predicted solutions for the angle (error_u).
The folder `continuous_time_identification’ corresponds to the results presented in Section III.C. By running the file swingEquation_identification.py we can predict system inertia and damping based on the input data (swingEquation_identification.mat). The exact values of the inertia and damping levels are 0.25 and 0.15. After the training process, the code prints the estimation error for the inertia (error_lambda_1) and damping (error_lambda_2), as well as the L2 error between exact and predicted solutions for the angle (error_u).

Code variables:
lb : defines the lower bound for the inputs (P,t)
ub: defines the upper bound for the inputs (P,t)
Nu : number of initial and boundary data
Nf : number of collocation points
usol (δ): is an array containing the angle trajectories for different pair of (P,t) (output to the NN)
x (P1): is an array containing different power levels in the range [0.08, 0.18] (input to the NN)
t : is an array containing time instants in the range [0, 20] (input to the NN)


When publishing results based on this data/code, please cite:
	G. Misyris, A. Venzke, S. Chatzivasileiadis, " Physics-Informed 
	Neural Networks for Power Systems", 2019. Available online: 
	https://arxiv.org/abs/1911.03737

@misc{misyris2019physicsinformed,
    title={Physics-Informed Neural Networks for Power Systems},
    author={George S. Misyris and Andreas Venzke and Spyros Chatzivasileiadis},
    year={2019},
    eprint={1911.03737},
    archivePrefix={arXiv},
    primaryClass={eess.SY}
}
