namespace Neuraval.ChatBot.Services
{
    public class LlamaCppOptions
    {
        public string BaseUrl { get; set; }
        public string Model { get; set; }
        public string ApiKey { get; set; }
        public double? Temperature { get; set; }
        public int? MaxTokens { get; set; }

        public LlamaCppOptions(
            string baseUrl = "http://localhost:8080/v1",
            string model = "default",
            string apiKey = "")
        {
            BaseUrl = baseUrl;
            Model = model;
            ApiKey = apiKey;
        }
    }
}
