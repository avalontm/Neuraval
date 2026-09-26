using System;

namespace Neuraval.ChatBot.Services
{
    public class OpenAiCompatibleOptions
    {
        public string BaseUrl { get; }
        public string Model { get; }
        public string ApiKey { get; set; }
        public double? Temperature { get; set; }
        public int? MaxTokens { get; set; }
        public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(100);

        public OpenAiCompatibleOptions(string baseUrl, string model, string apiKey = "")
        {
            if (string.IsNullOrWhiteSpace(baseUrl))
                throw new ArgumentException("baseUrl is required", nameof(baseUrl));

            if (string.IsNullOrWhiteSpace(model))
                throw new ArgumentException("model is required", nameof(model));

            BaseUrl = baseUrl.TrimEnd('/');
            Model = model;
            ApiKey = apiKey ?? "";
        }
    }
}
