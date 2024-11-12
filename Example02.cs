// Step 2, Kernel
// Chat Service
// Streaming Chat Completion Call

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

internal sealed partial class Examples
{
    internal async Task Example02()
    {
        var modelId = "llama3.2";
        var endpoint = new Uri("http://localhost:11434");

        var kernel = Kernel.CreateBuilder()
            .AddOllamaChatCompletion(modelId, endpoint)
            .Build();

        await foreach (var token in kernel.InvokePromptStreamingAsync("Hello, how are you?"))
        {
            Console.Write(token);
        };
    }
}