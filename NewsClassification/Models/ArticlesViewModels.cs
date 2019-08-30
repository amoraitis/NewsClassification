using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Newtonsoft.Json;

namespace NewsClassification.Models
{
    public class ArticlesViewModel
    {
        public IList<Article> Articles { get; set; } = new List<Article>();
        private static readonly string contentDirectory = Path.Combine(Environment.CurrentDirectory, "content");

        public static IList<Article> DeserializeData()
        {
            var newsCategoryJson = File.ReadAllText(Path.Combine(contentDirectory, @"News_Category_Dataset_v2.json"));
            return JsonConvert.DeserializeObject<List<Article>>(newsCategoryJson);
        }
    }

    public class Article
    {
        [LoadColumn(0)]
        public string Category { get; set; }
        [LoadColumn(1)]
        public string Headline { get; set; }
        [LoadColumn(2)]
        public string Authors { get; set; }
        [LoadColumn(3)]
        public string Link { get; set; }
        [LoadColumn(4)]
        public string short_description { get; set; }
        [LoadColumn(5)]
        public string Date { get; set; }
    }

    public class CategoryPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedCategory { get; set; }
    }
}
