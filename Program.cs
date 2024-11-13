using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureAIInference;
using Microsoft.SemanticKernel.Connectors.Ollama;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Services;
using OllamaSharp;
using System.ComponentModel;
using System.Text;

List<Func<Task>> examples = [];

#region Examples

examples.Add(async () => // Kernel prompting blocking (non-streaming)
{
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");

    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint)
        .Build();

    var response = await kernel.InvokePromptAsync("Hello, how are you?");

    Console.WriteLine(response);
});

examples.Add(async () => // Service prompting blocking (IChatClient)
{
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var service = new OllamaApiClient(endpoint, modelId)
        .AsChatCompletionService();

    var response = await service.GetChatMessageContentAsync("Hello, how are you?");
    Console.WriteLine(response);

    Console.WriteLine("\n----\n");

    // Normally usage of the service directly allows you to use specific modality types like
    // Sending the chat history.
    ChatHistory chatHistory = [
        new ChatMessageContent(AuthorRole.System, "You are presenting in a conference, and your name is Roger Bot."),
        new ChatMessageContent(AuthorRole.User, "Hello, how are you?")
    ];

    response = await service.GetChatMessageContentAsync(chatHistory);

    Console.WriteLine(response);
});

examples.Add(async () => // Kernel prompting streaming
{
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");

    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint)
        .Build();

    await foreach (var token in kernel.InvokePromptStreamingAsync("Why is Neptune blue?"))
    {
        Console.Write(token);
    };
});

examples.Add(async () => // Kernel prompting with custom settings
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

    await foreach (var token in kernel.InvokePromptStreamingAsync("Why does Jupiter has storms?", new(settings)))
    {
        Console.Write(token);
    };
});

examples.Add(async () => // Kernel prompting with Templated Prompt with Variables
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
    }
});

examples.Add(async () => // Kernel prompting with Templated Prompt with Function Plugins
{
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint)
        .Build();

    string GetTime()
    {
        return DateTime.UtcNow.ToString("R");
    }

    var myFunction = KernelFunctionFactory.CreateFromMethod(GetTime, "GetTime");
    kernel.Plugins.AddFromFunctions("MyPlugin", [myFunction]);

    var settings = new OllamaPromptExecutionSettings
    {
        TopP = 0.9f,
        Temperature = 0.9f,
    };
    var myPrompt = "Hello, I'm {{$name}}, and current time now is {{GetTime}}. What day is today?";
    var arguments = new KernelArguments(settings)
    {
        ["name"] = "Roger"
    };

    await foreach (var token in kernel.InvokePromptStreamingAsync(myPrompt, arguments))
    {
        Console.Write(token);
    }
});

examples.Add(async () => // Kernel prompting with Templated Prompt with Stateless Plugin
{
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernelBuilder = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint);

    // Plugins can be added also at the kernel configuration phase.
    kernelBuilder.Plugins.AddFromType<MyStatelessPlugin>();

    var kernel = kernelBuilder.Build();

    var settings = new OllamaPromptExecutionSettings
    {
        TopP = 0.9f,
        Temperature = 0.9f,
    };

    var myPrompt = "Hello, I'm {{$name}}, and current time now is {{GetTime}}. What day is today?";
    var arguments = new KernelArguments(settings)
    {
        ["name"] = "Roger"
    };

    await foreach (var token in kernel.InvokePromptStreamingAsync(myPrompt, arguments))
    {
        Console.Write(token);
    }
});

examples.Add(async () => // Kernel prompting with Templated Prompt with Stateful Plugins
{
    var myStatefulPlugin = new MyStatefulPlugin();
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint)
        .Build();

    kernel.Plugins.AddFromObject(myStatefulPlugin);
    var myPrompt = "Hello, I'm {{$name}}, and current count now is {{GetCounter}}. Is the current count an even or an odd number?";
    var arguments = new KernelArguments
    {
        ["name"] = "Roger"
    };

    await foreach (var token in kernel.InvokePromptStreamingAsync(myPrompt, arguments))
    {
        Console.Write(token);
    }
    Console.WriteLine("\n----\n");
    await foreach (var token in kernel.InvokePromptStreamingAsync(myPrompt, arguments))
    {
        Console.Write(token);
    }
});

examples.Add(async () => // Kernel prompting with function calling and stateful Plugins
{
    var myDescribedStatefulPlugin = new MyDescribedStatefulPlugin();
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernelBuilder = Kernel.CreateBuilder();
    
    kernelBuilder.Services.AddSingleton((sp) => 
    {
        return new ChatClientBuilder()
            .UseFunctionInvocation()
            .Use(new OllamaApiClient(endpoint, modelId))
            .AsChatCompletionService();
    });

    var kernel = kernelBuilder.Build();

    kernel.Plugins.AddFromObject(myDescribedStatefulPlugin);

    var myPrompt = "Hello, I'm {{$name}}. Is the current count an even or an odd number and what is its number?";
    var settings = new OllamaPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };
    var arguments = new KernelArguments(settings)
    {
        ["name"] = "Roger"
    };

    // Ollama only support function calling without streaming mode.
    Console.WriteLine("Processing...");
    var result = await kernel.InvokePromptAsync(myPrompt, arguments);
    Console.WriteLine(result);

    Console.WriteLine("Processing..."); 
    result = await kernel.InvokePromptAsync(myPrompt, arguments);
    Console.WriteLine(result);
});

examples.Add(async () => // Service prompting with function calling and stateful Plugins
{
    var myDescribedStatefulPlugin = new MyDescribedStatefulPlugin();
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernelBuilder = Kernel.CreateBuilder();

    var service = new ChatClientBuilder()
            .UseFunctionInvocation()
            .Use(new OllamaApiClient(endpoint, modelId))
            .AsChatCompletionService();

    var kernel = kernelBuilder.Build();
    kernel.Plugins.AddFromObject(myDescribedStatefulPlugin);

    var myPrompt = "Is the current count an even or an odd number and what is its number?";
    var settings = new OllamaPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };

    ChatHistory chatMessageContents = [
        new ChatMessageContent(AuthorRole.User, myPrompt)
    ];

    // Ollama only support function calling without streaming mode.
    Console.WriteLine("Processing...");
    var result = await service.GetChatMessageContentAsync(chatMessageContents, settings, kernel);
    Console.WriteLine(result);

    Console.WriteLine("Processing...");
    result = await service.GetChatMessageContentAsync(chatMessageContents, settings, kernel);
    Console.WriteLine(result);
});

examples.Add(async () => // Kernel with multiple chat completion services
{
    var ollamaModelId = "llama3.2";
    var ollamaEndpoint = new Uri("http://localhost:11434");
    var fileModelId = "phi3";
    var fileModelPath = "E:\\repo\\huggingface\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
    var kernelBuilder = Kernel.CreateBuilder();

    kernelBuilder
        .Services
            .AddSingleton((sp) => new OllamaApiClient(ollamaEndpoint, ollamaModelId).AsChatCompletionService())
            .AddOnnxRuntimeGenAIChatCompletion(fileModelId, fileModelPath);
            // Last service will be the default service

    var kernel = kernelBuilder.Build();

    kernel.PromptRenderFilters.Add(new SelectedModelRenderFilter());

    var modelNames = string.Join(", ", kernel.GetAllServices<IChatCompletionService>().Select(s => s.GetModelId()));

    while (true)
    {
        Console.Write($"\n\nType one of the available models [{modelNames}] > ");
        var selectedModel = Console.ReadLine();
        Console.WriteLine();

        if (string.IsNullOrEmpty(selectedModel)) { break; }

        var settings = new PromptExecutionSettings { ModelId = selectedModel };
        await foreach (var token in kernel.InvokePromptStreamingAsync("Answer in a small sentence, why does Jupiter have storms?", new(settings)))
        {
            Console.Write(token);
        };
    }
});

examples.Add(async () => // Multi modalities text -> audio services
{
    var config = new ConfigurationBuilder()
        .AddUserSecrets<Program>()
        .Build();

    var chatService = new AzureAIInferenceChatCompletionService(
        modelId: "phi3",
        apiKey: config["AzureAIInference:ApiKey"]!, 
        endpoint: new Uri(config["AzureAIInference:Endpoint"]!));

    var textToAudioService = new OpenAITextToAudioService(
        modelId: "tts-1", 
        apiKey: config["OpenAI:ApiKey"]!);

    StringBuilder answer = new();
    await foreach (var token in chatService.GetStreamingChatMessageContentsAsync("Explain in a simple phrase why Jupiter has storms."))
    {
        Console.Write(token);
        answer.Append(token);
    }

    Console.WriteLine("\nGenerating audio...\n");
    var generatedAudios = await textToAudioService.GetAudioContentsAsync(answer.ToString());
    var generatedAudio = generatedAudios[0];

    var file = Path.Combine(Directory.GetCurrentDirectory(), "output.mp3");
    await File.WriteAllBytesAsync(file, generatedAudio.Data!.Value);

    Console.WriteLine($"\nAudio Generated. Ctrl + Click to listen: {new Uri(file).AbsoluteUri}");
});

examples.Add(async () => // Multi modalities audio -> text services
{
    var config = new ConfigurationBuilder()
        .AddUserSecrets<Program>()
        .Build();

    string folderPath = @"C:\Users\rbarreto\OneDrive - Microsoft\Documents\Sound Recordings";

    DirectoryInfo directoryInfo = new DirectoryInfo(folderPath);

    // Get the most recent file created in the directory
    FileInfo? mostRecentFile = directoryInfo.GetFiles()
                                           .OrderByDescending(f => f.CreationTime)
                                           .FirstOrDefault();

    Console.WriteLine($"Most recent file: {mostRecentFile!.FullName}");

    var audioContent = new Microsoft.SemanticKernel.AudioContent(File.ReadAllBytes(mostRecentFile!.FullName), "audio/m4a");

    var audioToTextService = new OpenAIAudioToTextService(
    modelId: "whisper-1",
    apiKey: config["OpenAI:ApiKey"]!);

    Console.WriteLine("\nGenerating text from audio ...\n");
    var generatedTexts = await audioToTextService.GetTextContentsAsync(audioContent);
    var generatedText = generatedTexts[0];

    Console.WriteLine($"Text Generated: {generatedText}");
});

examples.Add(async () => // Multi modalities audio -> text -> image services
{
    var config = new ConfigurationBuilder()
        .AddUserSecrets<Program>()
        .Build();

    string folderPath = @"C:\Users\rbarreto\OneDrive - Microsoft\Documents\Sound Recordings";

    DirectoryInfo directoryInfo = new DirectoryInfo(folderPath);

    // Get the most recent file created in the directory
    FileInfo? mostRecentFile = directoryInfo.GetFiles()
                                           .OrderByDescending(f => f.CreationTime)
                                           .FirstOrDefault();

    Console.WriteLine($"Most recent file: {mostRecentFile!.FullName}");

    var audioContent = new Microsoft.SemanticKernel.AudioContent(File.ReadAllBytes(mostRecentFile!.FullName), "audio/m4a");

    var audioToTextService = new OpenAIAudioToTextService(
        modelId: "whisper-1",
        apiKey: config["OpenAI:ApiKey"]!);

    var textToImageService = new OpenAITextToImageService(
    modelId: "dall-e-3",
    apiKey: config["OpenAI:ApiKey"]!);

    Console.WriteLine("\nGenerating text from audio ...\n");
    var generatedTexts = await audioToTextService.GetTextContentsAsync(audioContent);
    var generatedText = generatedTexts[0];

    Console.WriteLine($"Text Generated: {generatedText}");

    Console.WriteLine("\nGenerating image from text ...\n");
    var generatedImages = await textToImageService.GetImageContentsAsync(generatedText);
    var generatedImage = generatedImages[0];

    Console.WriteLine("Image Generated. Ctrl + Click to view: " + generatedImage.Uri);
});

#endregion Examples

await examples[^1](); // Run the last example

Console.WriteLine("\n\n\n");

#region Plugins

public class MyStatelessPlugin
{
    public string Name => "MyPlugin";

    [KernelFunction]
    public string GetTime()
    {
        return DateTime.UtcNow.ToString("R");
    }
}

public class MyStatefulPlugin
{
    private int _counter = 0;

    [KernelFunction]
    public int GetCounter()
    {
        return _counter++;
    }
}

public class MyDescribedStatefulPlugin
{
    private int _counter = 0;

    [KernelFunction, Description("Current count number")]
    [return: Description("The current count number")]
    public int GetCounter()
    {
        return _counter++;
    }
}

#endregion

#region Filters

public class SelectedModelRenderFilter : IPromptRenderFilter
{
    public async Task OnPromptRenderAsync(PromptRenderContext context, Func<PromptRenderContext, Task> next)
    {
        Console.WriteLine($"Settings model id: {context.Arguments.ExecutionSettings?.FirstOrDefault().Value.ModelId}");

        await next(context);
    }
}
#endregion

#region Debugging
public class CustomHttpClientHandler : HttpClientHandler
{
    protected async override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        // Console.WriteLine($"Request: {request.RequestUri}");
        var requestBody = await request.Content!.ReadAsStringAsync();
        // Console.WriteLine($"Request Body: {requestBody}");
        return await base.SendAsync(request, cancellationToken);
    }
}
#endregion