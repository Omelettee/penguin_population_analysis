## Penguin Population Analysis
  - This project is designed to reveal the relationship between penguin population and environmental factors
<br></br>
### 1. DESCRIPTION - Describe the package in a few paragraphs.
The model code is contained within CODE/data/Model
The model.py is written with Python, the packages are listed following:
  - from pandas import DataFrame
  - import pandas as pd
  - from matplotlib import pyplot
  - import random
  - from scipy.stats.stats import pearsonr
  - import matplotlib.pyplot as plt
  - from sklearn.model_selection import cross_validate, train_test_split,GridSea
  - from sklearn.metrics import accuracy_score 
  - from sklearn.linear_model import LinearRegression
<br></br>


### 2. Implement Model
Run the model.py in the folder of “model” with command :
- `python  model.py`
- All the data needed is stored in the folder “data_original” \- the same layer with model.py
<br></br>
### 3. Implement Visualization
The visualization is contained within directory CODE.
To run visualization:
- 1. Install QGIS 3.0 software from https://qgis.org/en/site/ to run this visualization. This is a scientific plotting software that is created for visualizing map and model data across many different sources.
- 2. Select AntarcticDataMap_QGIS3.qgs
- 3. Select desired layer from "gentoo" "adellie" "emperor" or "chinstrap" to view by choosing the radio button in the Layer pallette.
- 4. To choose a year:
   - Right click on layer in the layer pallete.
   - In the dialog box, edit the current year to the desired year.
   - If you want to view a range of years you can use this command:
   -   ` "year" = 1990 AND "year" = 2000`
   - Each layer must be edited individually.
 - 5. Additional layers from ADD can be sourced using the following:
    - Use ADD to map with QGIS database program and import using the WMS/WMTS layer add menu option.
    - Use the following URL to connect and add layers: https://maps.bas.ac.uk/wmts?request=GetCapabilities
    - Add resource: https://add.data.bas.ac.uk/repository/entry/show?entryid=f477219b-9121-44d6-afa6-d8552762dc45&ascending=true&orderby=createdate&showentryselectform=true
      \(This is similar to the layers added on here: https://www.add.scar.org/)

