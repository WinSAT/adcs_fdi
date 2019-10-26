# Fault Isolation of Reaction Wheels Onboard 3-axis Controlled In-orbit Satellites Using Ensemble Machine Learning Techniques

The primary objective of this study is to explore novel applications of data-driven machine learning methods for isolation of nonlinear systems with a case-study for an in-orbit closed-loop controlled satellite with reaction wheels as actuators. High-fidelity models of the 3-axis controlled satellite are developed to provide an abundance of data for both healthy and various faulty conditions of the satellite. This data is then used as input for the proposed data-driven fault isolation method. Once a fault is detected, the fault isolation module is activated where it employs a machine learning technique which incorporates ensemble methods involving random forests, decision trees, and nearest neighbors. Results of the classified faulty condition are then cross-validated using k-fold and leave-one-out methods. A comprehensive comparison of the performance of different combinations for the ensemble architecture. Results show promising outcomes for fault isolation of the non-linear systems using ensemble methods.

You can also view the [Conference Paper](https://www.researchgate.net/publication/336106740_Fault_isolation_of_reaction_wheels_onboard_3-axis_controlled_in-orbit_satellite_using_ensemble_machine_learning_techniques) and [Conference Presentation](https://www.researchgate.net/publication/336106932_Fault_Isolation_of_Reaction_Wheels_Onboard_3-axis_Controlled_In-orbit_Satellites_Using_Ensemble_Machine_Learning_Techniques) submitted for [ICASSE 2019](http://www.utias.utoronto.ca/icasse/)

## How to install/run ADCS FDI model (python 2.7)
May need to install [Hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) if needed
```
#install needed python packages
pip install --upgrade pip
pip install numpy scipy matplotlib pandas sklearn
pip install tsfresh hyperopt xgboost lightgbm

#clone and run
git clone https://github.com/WinSAT/adcs_fdi.git
cd ./adcs_fdi
python adcs_fdi.py
```

Task List
- [x] Provide initial trained model for ICASSE
- [ ] Optimize hyperparameters, feature sets for optimal classification
- [ ] Implement ADCS into [WinSAT-1 cubesat](https://winsat.ca)