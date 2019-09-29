using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using NewsClassification.Models;

namespace NewsClassification.ML
{
    public class MLTrain
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _modelPath => Path.Combine(Directory.GetParent(_appPath).Parent.Parent.FullName, "Models", "model.zip");
        private MLContext _mlContext;
        private ITransformer _trainedModel;
        private PredictionEngine<Article, CategoryPrediction> _predictionEngine;

        public MLTrain(bool withTraining)
        {
            if (withTraining)
                TrainData();
            else
                _mlContext = new MLContext();
        }

        public void TrainData()
        {
            // Create MLContext
            _mlContext = new MLContext();

            var articlesList = ArticlesViewModel.DeserializeData();
            var data = _mlContext.Data.LoadFromEnumerable<Article>(articlesList);

            var dataSplit = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;
            var pipeline = ProcessData();
            var trainStartedDateTime = DateTime.Now;
            var trainingPipeline = BuildAndTrain(trainData, pipeline);
            var trainEndedDateTime = DateTime.Now;
            var span = trainEndedDateTime - trainStartedDateTime;
            var totalMinutes = span.TotalMinutes;
            Console.WriteLine($"Total minutes to train with 2 iterations: {totalMinutes:N3}");
            Evaluate(trainData.Schema, testData);
        }
        private IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms
                
                .Conversion.MapValueToKey(inputColumnName:nameof(Article.Category), outputColumnName:"Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.Headline),
                    outputColumnName: "HeadlineFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Article.short_description),
                    outputColumnName: "short_descriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "HeadlineFeaturized",
                    "short_descriptionFeaturized"));

            return pipeline;
        }
        

        private IEstimator<ITransformer> BuildAndTrain(IDataView trainData, IEstimator<ITransformer> pipeline)
        {
            // Create the training algorithm class
            // Alternative SdcaMaximumEntropy
            // Create a binary classification trainer.
            var averagedPerceptronBinaryTrainer = _mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 2);
            // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
            // In this strategy, a binary classification algorithm is used to train one classifier for each class, "
            // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
            // and choosing the prediction with the highest confidence score.
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            _trainedModel = trainingPipeline.Fit(trainData);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<Article, CategoryPrediction>(_trainedModel);
            
            // Return the model
            return trainingPipeline;
        }

        internal void PredictArticle(Article singleArticle)
        {
            var loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<Article, CategoryPrediction>(loadedModel);

            var prediction = _predictionEngine.Predict(singleArticle);

            FullPrediction[] predictions;
            float[] scores = prediction.Score;
            int size = scores.Length;
            int index0, index1, index2 = 0;

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            _predictionEngine.OutputSchema[nameof(CategoryPrediction.Score)].GetSlotNames(ref slotNames);

            GetIndexesOfTopThreeScores(scores, size, out index0, out index1, out index2);

            predictions = new FullPrediction[]
            {
                new FullPrediction(slotNames.GetItemOrDefault(index0).ToString(),scores[index0],index0),
                new FullPrediction(slotNames.GetItemOrDefault(index1).ToString(),scores[index1],index1),
                new FullPrediction(slotNames.GetItemOrDefault(index2).ToString(),scores[index2],index2)
            };

            Console.WriteLine("Predicted the given article's category:");
            Console.WriteLine($"- With title: {singleArticle.Headline}\n");
            Console.WriteLine($"- With description: {singleArticle.short_description}\n");
            Console.WriteLine($"=============== and prediction: {prediction.PredictedCategory} ===============");
        }

        private void GetIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)
        {
            int i;
            float first, second, third;
            index0 = index1 = index2 = 0;
            if (n < 3)
            {
                Console.WriteLine("Invalid Input");
                return;
            }
            third = first = second = 000;
            for (i = 0; i < n; i++)
            {
                // If current element is  
                // smaller than first 
                if (scores[i] > first)
                {
                    third = second;
                    second = first;
                    first = scores[i];
                }
                // If arr[i] is in between first 
                // and second then update second 
                else if (scores[i] > second)
                {
                    third = second;
                    second = scores[i];
                }

                else if (scores[i] > third)
                    third = scores[i];
            }
            var scoresList = scores.ToList();
            index0 = scoresList.IndexOf(first);
            index1 = scoresList.IndexOf(second);
            index2 = scoresList.IndexOf(third);
        }

        private void Evaluate(DataViewSchema trainingDataViewSchema, IDataView testData)
        {
            // Create the multiclass evaluator
            // Evaluate the model and create metrics
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testData));


            // Display the metrics
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        private void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        

       
    }
}
