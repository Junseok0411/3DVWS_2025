# Exploring in silico pharmacological actions of pyramax against Sars-cov-2



## Main contents
- Pyronaridine & Artesunate pharmacological actions against SARS-CoV-2

## Introduction
This study describes the basic steps and methodologies used in ‘in silico’ drug discovery. Based on the in vitro evidence that Pyramax inhibits the replication of SARS-CoV-2, this study systematically investigates the pharmacological potential of Pyramax using an ‘in silico’ approach.

## Drug discovery in silico
![image](https://github.com/user-attachments/assets/8caec878-aec9-4f05-aa0a-fcd36252176b)
- With computational methods, ligands 3D modeling and induced-fit simulations can be conducted. By doing so, it is possible to check the potential possibility of the drugs against the pathogen.

## 6M0J
![6M0J](https://github.com/user-attachments/assets/29649489-8a54-4bee-b6a1-31392da2c9a3)
It is a crystal strucure of SARS-CoV-2 spike receptor-binding domain bound with ACE2.
- Orange colored protein is Spike protein S1 from SARS-CoV-2
- Green colored protein is ACE2 enzyme from human


## Pyramax
There are several studies that pyramax inhibits the replication of SARS-CoV-2 _in vitro_ evaluation.
- Artesunate
![artesunate](https://github.com/user-attachments/assets/b5bae163-6da3-4f7c-9c0d-da6cc95d6415)

- Pyronaridine
![pyronaridine](https://github.com/user-attachments/assets/33438295-e2f7-49d4-badb-0aeaf365e3f5)

## Workflow
![Workflow](https://github.com/user-attachments/assets/d5b35065-f35a-411e-b803-3e46e4af862e)

## AutoDock: Binding affinity
![Result of the binding affinity](https://github.com/user-attachments/assets/ed54bedc-c1dd-4d40-8786-fa7d9430a8e2)
Select the models that have the lowest value of affinity, determining the best site and the best complex model.

Criteria for the binding affinity
![Criteria](https://github.com/user-attachments/assets/7d4692e8-2d78-4b23-bd09-759dba71d1a1)

## GROMACS
conditions: 310 K, 1 atm
![image](https://github.com/user-attachments/assets/a6cf2c2c-1f6d-4eb6-b768-8d90f842db59)

- Energy Minimization
  Process of minimizing the potential energy of a system to increase structural stability.
![Energy](https://github.com/user-attachments/assets/a3f6c41c-5f69-447a-901c-68b2da5d098d)

-  RMSD simulation
  RMSD = Root Mean Square Deviation compared with the initial site value
![RMSD](https://github.com/user-attachments/assets/db3a2059-58e8-4f4d-b930-acf71d58aa04)
At initial state, the RMSD value is dramatically increased.
However, after 1500 ps, the complex structure get stable value within 0.05nm to 0.06nm. It means the complex get significantly stable.





