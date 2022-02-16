# HLA registry Epitopes Distances to the cell-membrane


### Core files:
- epitope_location.ipynb: This notebook entails all the details and scripts on finding the location of the Epitopes per HLA allel. 
- epitope_distance2membrane: This notebook entails all the details and scripts on finding the distances of the Epitopes w.r.t the tail of the corresponding HLA to whcih they belong (cell-membrane ).

#### Meda Data:
The above scripts lead to adding new variables to the Epitope Database. Such variables as 
- Location in x, y, z
- Distance: The distances of the epitope to the tail of each HLA to which it belongs in Angestrom [A]. This is called distance to the cell-membrane.
- Mean_distance [A]: The mean of distance variable among the HLA's. 
- std_distance [A]: The standard-deviation of distances to see if an epitope location is very different among all the HLA molecules.

### Analysis files:
The other files starting with analysis are basically analysing the Epitope Registry database available in pickle & csv (file name EpitopevsHLA) in this directory.