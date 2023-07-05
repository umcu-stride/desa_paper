# desa_paper
---

This repository entails the most recent version of the scripts used in STRIDE project with the aim of delivering the analysis required for the DESA paper. 


### Components 
---

The directory components are sctructured as sources codes, i.e. `/src` and `Notebooks`, where the source codes are all called into relevant Notebooks. 

* Source codes `/src`entail:
    * `pipe_store`: All the required methods for preprocessing consumed by pandas `pipe` method.
    * `helper_methods`: All the required methods to carry out analysis or visualisation. Priliminary version of these scripts are also available in `stride_analytics` repository for the sakee of completeness. However, it is advised to consider the methods in this repository to be more mature. 

    * `utility` & `constants`: All the utility methods and constants used either in `pipe_store` or `helper_methods`

    * `/leuven`: Leuven directory inculdes all the scripts to preprocess (clean, structure, ...) the Leuven data and find DSA and DESA by customising the antibody analysis pipeline that were used in STRIDE project. The original STRIDE antibody analysis pipeline is available in `stride_analytics` repository. 


* `/Notebooks` entails:

    * Antibody_Eptiopes analysis: Some visualisation of the ellipro scores and mfi values for early and late grat losses.

    * Cox proportional hazard analysis: CPH analysis is carried out for differtent features in the data set (bad desa vs other/good desa ...).

    * dataset charachteristics: This is basically aiming at creating charachteristic tables by replicating the R `tableone` method in python.  

    * epitope distances: The scripts to visualise the epitope distance distribution. These scripts are borrowed from `epitopre-databse` repository. Therefore, for more information, I refer you to  `epitope_distances/README.md` file and `epitopre-databse` repository.

    * epitope relevance: Kaplan-Meier visualisation for different variables including, good/bad (relevant/irrelevant) DESA, desa recognised by mAbs, and DESA quantity. 
        - The investigation-validation folder entails the latest analysis for the final manuscript vs the old analysis 

    * Leuven Data: Consumes the methods in the `src/leuven` to find antibody information, i.e. DSA & DESA, in Leuven data. There are are quite some txt file on the content of the communications with Alek Senev and scripts (`lsa_data_preparation`) to clean, prepare, and structure the LSA data to be later used by the antibody analysis pipeline. The data literacy on the Leuven data is quite limited at the time of writing this note.


# Creating virtual environments with pipenv:
---

- Pipfile: for making local virtual environments for installing python packages. To produce the virtual environment, below steps are required:

    * Install the pipenv package `pip install --user pipenv`
    * Create the virtual environment `pipenv install --python 3.8`. (Pipfile packages will automatically install)
    * Activate the virtual environment with `pipenv shell`
    * Make sure the python files & notebooks are hooked up to the virtual environment before running the scripts