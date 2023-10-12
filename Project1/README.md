# FYS-STK4155: Project 1:


## Project overview:
This is the code for reproducing our results in _Project 1_ of **FYS-STK4155** for the Autumn 2023 semester at UiO.


## Installation Instructions:
To install all the nesecarry packages run this code here:

```Python
pip install -r requirements.txt
```

where **requirements.txt** contains all the required packages do run the code for this repository.


## Datasets:
There are two dataset used here, one from real data and one from syntizhed data.

### Syntizhed data:
We use a syntizhed dataset that is made using the _Franke function_. The function `def FrankeFunction` is defined in **reg_class.py**.

### Real data:
The dataset is digital terrain data of Norway. It is the file **SRTM_data_Norway_1.tif** and can be found at:

[https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project1/DataFiles/SRTM_data_Norway_1.tif
](https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project1/DataFiles/SRTM_data_Norway_1.tif
)



## Usage Guide:
If you run the file **topological.py**, you run the analysis on our dataset:

```Python
python3 topological.py
```

while if you run the file **reg_class.py** as a script:

```Python
python3 reg_class.py
```

it will be on the synthized dataset (made from the _Franke function_).

The files **cross_val.py** and **cross_val_debug.py** was created for developing and debugging the cross validation code in **reg_class.py** and are irrelevant for the analysis. However to run the other resampling method, _bootstrap_, you need to run:

```Python
python3 ec3.py
```

Lastly **test_reg_class.py** is where we have tested that our code ran correctly. To run these tests write this in the terminal:

```Python
python3 test_reg_class.py
```