# Green roofs: automatic detection of roof vegetation, vegetation type and covered surface

This project provides a suite of Python scripts allowing the end-user to use machine learning to detect green roofs on land survey building footprint based on orthophotos. 

## Hardware requirements

No specific requirements. 

## Software Requirements

* Python 3.9: The dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file. The following method was tested successfully on a Windows system: 

    ```bash
    $ conda create -n <the name of the virtual env> -c conda-forge python=3.9 gdal
    $ conda activate <the name of the virtual env>
    $ pip install -r setup/requirements.txt
    ```

## Folder structure

```
├── config                        # config files
├── data                          # data to process, see addendum
├───scripts
│   │   calculate_raster.py       # compute the NDVI and luminosity raster of the orthoimage tiles 
│   |   clip_image.py             # clip the orthoimages for the aoi extent. 
│   |   greenery.py               # main workflow, greenery detection and logistic regression
│   |   log_reg.py                # workflow for logistic regression only (ex. from intermediate results of greenery.py)
│   |   roof_stats.py             # scripts to study and prepare the ground truth layer
│   |   
│   └───functions                 # set of functions used in python scripts
└── setup                         # requirements for environment installation
```

## Scripts and Procedure

The following abbreviations are used:

* **AOI**: area of interest

* **GT**: ground truth

* **LR**: logistic regression

Scripts are run in combination with hard-coded configuration files in the following order: 

1. `clip_image.py`
2. `calculate_raster.py`
3. `roof_stats.py`
4. `greenery.py`
5. `log_reg.py`


## Data preparation
1. `clip_image.py`: The goal of this script is to clip images with a AOI vector layer. In a first step, the AOI is buffered by 50 m. This vector layer is then used as an input to clip aerial imagery data.
	* Use clip_image.yaml to specify the inputs data. 
2. `calculate_raster.py`: compute NDVI and luminosity rasters. Watch out for the right band in functions `calculate_ndvi` and `calculate_luminosity`. 
	* Use logReg.yaml to specify the inputs data.
3. `roof_stats.py`: compute statistics of NDVI and luminostiy values per roofs to help define thresholds. Split the roofs into a training and a test dataset. 
	* Use logReg.yaml to specify the inputs data.
	* Please verifiy that the join option ("predicate") in [`functions/fct_misc.py`](./scripts/functions/fct_misc.py) in Line 83 is "within".

## Logistic regression approach
The logistic regression approach was developed inspired by Louis-Lucas et al (1) and implemented for the specific project in `functions/fct_misc.py`. 

4. `greenery.py`: identify greenery on roofs based on NDVI values and luminosity to make a selection of roofs before training a logistic regression. 
	* Use logReg.yaml to specify the inputs data.

5. `log_reg`: focuses on the logistic regression part of the pipeline.
	* Use logReg.yaml to specify the inputs data.



## Addendum

### Documentation
The full documentation of the project is available on the [STDL's technical website](https://tech.stdl.ch/PROJ-VEGROOFS/).

### Data 

#### Ground truth 

The ground truth consists of ...
* Labelling of ground truth by the beneficiaries (Februar 2024)


#### Folder structure 
Here following a proposition of data structure.

```
├── data                          # dataset folder
   ├── 01_initial                 # initial data 
      ├── AOI                     # AOI shape file
      ├── ground_truth            # ground truth shape file
      └── scratch                 # aerial images
           ├── extent             # tile extent computed at the beginning of the workflow
           └── tiles              # image tiles
   ├── 02_intermediate            # intermediate results and processed data
      ├── th                      # hydra documentation of values tested for the thresholds. 
      └── images
            ├── tiles             # clipped images
            ├── extent            # clipped tile extent 
            ├── luminosity        # luminosity tiles computed from NirRGB tiles
            └── ndvi              # NDVI tiles computed from NirRGB tiles
   └── 03_results                 # results of the workflows (training and test partition)
      └── scratch                 # roof stats, boxplots
```

### References
[1] Louis-Lucas, Tanguy, Flavie Mayrand, Philippe Clergeau, and Nathalie Machon. “Remote Sensing for Assessing Vegetated Roofs with a New Replicable Method in Paris, France.” Journal of Applied Remote Sensing 15, no. 1 (January 2021): 014501. https://doi.org/10.1117/1.JRS.15.014501.
