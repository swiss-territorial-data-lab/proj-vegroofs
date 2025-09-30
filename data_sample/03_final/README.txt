- scratch_greenery: 
-- roof_stats.csv: descriptors and ground truth for the training, statistics of the pixels per roof, image source "scratch" images**. 
-- *.pkl: trained models

- scratch_roof:
-- roof_stats.csv: descriptors and ground truth for the training, statistics of the pixels per potential greenery surface, image source "scratch" images.
-- *.pkl: trained models 

- gt.tot.gpkg: data to train and test the model. Labels were assigned manually by visually evaluating the rooftop condition on the image. There is no guaranty of the data to be error free.
--- veg_new_3: green or not
--- class: multiclass label 


** Attached tables and models were not obtained by using SWISSIMAGE RS, but a processed version of these images that changed the 16-bits encoding into 8-bits, mentioned here as "scratch". 