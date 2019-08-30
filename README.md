## NewsClassification

Using ML.NET to classify articles, based on title and description.</br>
Data source: https://www.kaggle.com/rmisra/news-category-dataset and changed the .json to an array of objects, for easy parsing with [Newtonsoft.Json](https://www.nuget.org/packages/Newtonsoft.Json/)

#### SdcaMaximumEntropy
*************************************************************************************************************
*       Metrics for Multi-class Classification model - Test Data
*------------------------------------------------------------------------------------------------------------
*       MicroAccuracy:    0.364
*       MacroAccuracy:    0.262
*       LogLoss:          3.858
*       LogLossReduction: -.177
*************************************************************************************************************


#### OVA (One-Versus-All) trainer with the BinaryTrainer.

###### Total minutes to train with 2 iterations: 7.545 mins
*************************************************************************************************************
*       Metrics for Multi-class Classification model - Test Data
*------------------------------------------------------------------------------------------------------------
*       MicroAccuracy:    0.58
*       MacroAccuracy:    0.427
*       LogLoss:          1.569
*       LogLossReduction: .522
***************************************************************************************************

##### 12 iterations(31 minutes to train the model)
*************************************************************************************************************
*       Metrics for Multi-class Classification model - Test Data
*------------------------------------------------------------------------------------------------------------
*       MicroAccuracy:    0.585
*       MacroAccuracy:    0.439
*       LogLoss:          1.873
*       LogLossReduction: .429
*************************************************************************************************************