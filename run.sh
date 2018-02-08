R CMD BATCH newPreprocess.R # data preproces and feature engineer for DL training
python3 modelPrepare.py     # prepare for model training
python3 M3Model.py          # the model training
python3 reloadModel.py      # reload the param file based on the loss of validation data
