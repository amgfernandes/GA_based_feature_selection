# GA_based_feature_selection

### Genetic algorithms (GA) Feature Selection based on sklearn-genetic-opt 
https://sklearn-genetic-opt.readthedocs.io/en/stable/index.html#sklearn-genetic-opt

Parser for command-line options is implemented

### Install

Example with new environment named `feature_selection`

```
conda create -n feature_selection python=3.9 -y

conda activate feature_selection

conda install pip -y

pip install -r requirements.txt
 ```

Script: GA_based_selection.py

Run in the terminal: 

- For help:
`python GA_based_selection.py -h`

- Run with: 
`python GA_based_selection.py` with the appropriate arguments

Example:

```
python GA_based_selection.py -g 5 -p 10 -c 0.2 -m 12
```

### Arguments:
```
'--generations', '-g', default=5
'--population_size', '-p', default=8
'--crossover_probability', '-c', default=0.1
'--max_features', '-m', default=10
```
###  Notes:

A log file and csv files are generated with parameters and selected features. Some plots for evaluation are also created.
