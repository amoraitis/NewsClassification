using System;
using NewsClassification.ML;
using NewsClassification.Models;

namespace NewsClassification
{
    internal class Program
    {
        private static readonly Article Article = new Article
        {
            Headline = "Biden Poses What-If Of Obama Assassination To Explain How 1968 Jolted America",
            short_description = "The former vice president told students at a presidential campaign event how the deaths of Bobby Kennedy and Martin Luther King Jr. led to his political awakening."
        };
        public static void Main(string[] args)
    {
        PredictArticleCategory();
        Console.ReadKey();
    }

    private static void PredictArticleCategory()
    {
        MLTrain mlTrain;
        try
        {
            mlTrain = new MLTrain(false);
            mlTrain.PredictArticle(Article);
        }
        catch // We don't already have a built model
        {
            // Go train a new one
            mlTrain = new MLTrain(true);
            mlTrain.PredictArticle(Article);
        }
    }
}
}
