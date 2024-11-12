using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;

internal sealed partial class Examples
{
    internal async Task Example04()
    {
        var modelId = "llama3.2";
        var endpoint = new Uri("http://localhost:11434");

        var kernel = Kernel.CreateBuilder()
            .AddOllamaChatCompletion(modelId, endpoint)
            .Build();

        var settings = new OllamaPromptExecutionSettings
        {
            TopP = 0.9f,
            Temperature = 0.9f,
        };

        var myPrompt = "Hello, I'm {{$name}}. What is my name?";

        var arguments = new KernelArguments(settings)
        {
            ["name"] = "Roger"
        };

        await foreach (var token in kernel.InvokePromptStreamingAsync(myPrompt, arguments))
        {
            Console.Write(token);
        };
    }
}