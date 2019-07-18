This repository contains accompanying code connected to across-sensor feature learning framework.

* `histories` are npy batches with `[loss, val_loss, acc, val_acc]` at each timestep`

* `models` are keras saved models

* `reports` are reports (statistics), generated from the dataset


Pipeline 

1. Download the data by shl_downloader.sh. Then **UNPACK archives like this:** 
    ```
    ├── src
    │   ├── data         
    │   │   ├── raw             
    │   │   │   ├── User1    
    │   │   │   │   ├── 220617 
    │   │   │   │   ├── 260617  
    │   │   │   │   └── 270617   
    │   │   │   ├── User2      
    │   │   │   │   ├── 140617    
    │   │   │   │   ├── 140717   
    │   │   │   │   └── 180717  
    │   │   │   └── User3  
    │   │   │       ├── 030717 
    │   │   │       ├── 070717  
    │   │   │       └── 140617   
    ```
    The only files you need in folders are: "Hips_motion.txt" and "Label.txt"
2. Preprocess data into batches of npy or csv by make_dataset.py
3. Run Classifier K to get classifier on K sensors
4. Run Classifier_single sensor to get many classifiers, one for each sensor
5. Run Duplex_Autoencoder_Training to train autoencoders
6. Run duplex_feature_extraction to translate data into features and save to the disk
7. Run duplex classifier to train classifiers on duplex features
8. The pipline results in "models" and "histoies" directories