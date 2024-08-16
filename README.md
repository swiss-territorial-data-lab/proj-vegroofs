# Green roofs: automatic detection of roof vegetation <!--, vegetation type and covered surface-->

This project provides a suite of Python scripts allowing the end-user to use machine learning to detect green roofs on land survey building footprint based on orthophotos. 

## Hardware requirements

No specific requirements. 

## Software requirements

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
│   │   calculate_raster.py       # computes the NDVI and luminosity rasters of the orthoimage tiles 
│   |   clip_image.py             # clips the orthoimages for the aoi extent 
│   |   greenery.py               # potential greenery detection by applying threshold on NDVI and luminosity rasters
│   |   infere_ml.py              # infers with the trained machine learning algorithms
│   |   train_ml.py               # trains and tests machine learning algorithms (logistic regression or random forest)
│   |   roof_stats.py             # computes the descriptors for the machine learning algorithms
│   |   
│   └───functions                 # set of functions used in Python scripts
└── setup                         # requirements for environment installation
```

## Scripts and procedure

The following abbreviations are used:

* **AOI**: area of interest
* **GT**: ground truth
* **LR**: logistic regression
* **RF**: random forest

Scripts are run in combination with hard-coded configuration files in the following order: 

1. `clip_image.py`
2. `calculate_raster.py`
3. `greenery.py`
4. `roof_stats.py`
5. `train_ml.py`
6. `infere_ml.py`

### Input data 

#### Ground truth 

The ground truth consists of a vector layer with the geometry of buildings from the land survey. Each building has a unique identifier, a label `green_tag` "green or not" and a class of vegetation type `green_cls` : bare (b), terrace (t), spontaneous (s), extensive (e), lawn (l) or intensive (i).

#### Images

Images should be NRGB. If the band order is different, please edit `calculate_raster.py` and adjust the band ordering in `roof_stats.py`.


## Data preparation
1. `clip_image.py`: The goal of this script is to clip images with a AOI vector layer. In a first step, the AOI is buffered by 50 m. This vector layer is then used as an input to clip aerial imagery data.
	* Use `clip_image.yaml` to specify the inputs data. 
2. `calculate_raster.py`: computes NDVI and luminosity rasters. Watch out for the right band numbering in functions `calculate_ndvi` and `calculate_luminosity`. 
	* Use `logReg.yaml` to specify the inputs and outputs directories.
      * ortho_directory
      * ndvi_directory
      * lum_directory

Steps (3) and (4) are about preparing the descriptors for the ML algorithms. `greenery.py` produces a polygon vector layer of potential greenery on roofs based on NDVI and luminosity values, and computes potential greenery ratio per roofs. This script is optional, because one may want to compute descriptors in (4) on the entire roofs and not on the potential green parts of the roofs.
 * Use `logReg.yaml` to specify the common inputs data to `greenery.py`and to `roofs_stats.py`.
      * tile_delimitation
      * gt
      * green_tag
      * green_cls
      * chm_layer
      * results_directory
      * egid_train_test
      * th_ndvi
      * th_lum 
      * epsg
3. `greenery.py`: produces a polygon vector layer of potential greenery on roofs based on NDVI and luminosity values, and computes potential greenery ratio per roofs. This script is optional. One may want to compute descriptors on the entire roof and not on the potential green parts of the roofs.
	* Use `logReg.yaml` to specify the inputs data.
      * hydra:run:dir
      * roofs_file
      * roofs_layer

4. `roof_stats.py`: computes statistics of NDVI and luminosity values per roofs. Splits the roofs into a training and a test dataset. 
	* Use`logReg.yaml` to specify the inputs data.
      * roofs_file
      * roofs_layer


## Machine learning
The machine learning approach was inspired by Louis-Lucas et al. (1) and adapted for the specificity of the project. In between, the machine learning algorithms and the descriptors used became rather different. 
 * Use `logReg.yaml` to specify the common parameters of the model to train and infer with in `train_ml.py` and in `infer_ml.py` respectively.

      * cls_ml
      * model_ml
      * trained_model_dir

1. `train_ml.py`: trains a logistic regression or a random forest and evaluates the trained algorithm on a test dataset. 
	* Use `logReg.yaml` to specify the inputs data.
         * roofs_file
         * roofs_layer
2. `infer_ml.py`: infers for descriptors computed with `roof_stats.py`. 
      * Use `logReg.yaml` to specify the inputs data.    
        * roofs_file
        * roofs_layer

## Addendum

### Documentation
The full documentation of the project is available on the [STDL's technical website](https://tech.stdl.ch/PROJ-VEGROOFS/).

### Folder structure 
Here following a proposition of data structure.

```
├── data                          # dataset folder
   ├── 01_initial                 # initial data 
      ├── aoi                     # AOI shape file
      ├── gt                      # ground truth shape file
      └── images                  # aerial images
           ├── extent             # tile extent computed at the beginning of the workflow
           └── tiles              # image tiles
   ├── 02_intermediate            # intermediate results and processed data
      ├── th                      # hydra timestamp folders for the tested thresholds. 
      └── images
            ├── tiles             # clipped images
            ├── extent            # clipped tile extent 
            ├── luminosity        # luminosity tiles computed from NirRGB tiles
            └── ndvi              # NDVI tiles computed from NirRGB tiles
   └── 03_results                 # results of the workflows (training and test partition)
      └── image_gt                # roof stats, boxplots, machine learning outputs on GT
      └── image_inf               # roof stats and machine learning outputs for inferences
```

### References
[1] Louis-Lucas, Tanguy, Flavie Mayrand, Philippe Clergeau, and Nathalie Machon. “Remote Sensing for Assessing Vegetated Roofs with a New Replicable Method in Paris, France.” Journal of Applied Remote Sensing 15, no. 1 (January 2021): 014501. https://doi.org/10.1117/1.JRS.15.014501.
