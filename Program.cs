using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Microsoft.SemanticKernel.Connectors.Ollama;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Data;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Services;
using OllamaSharp;
using OpenTelemetry;
using OpenTelemetry.Logs;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using System;
using System.ComponentModel;
using System.Text;

List<Func<Task>> examples = [];
var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();

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

examples.Add(async () => // Service prompting blocking (IChatClient Microsoft AI Abstractions)
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
    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId)
        .Build();

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

examples.Add(async () => // Get Service from Kernel prompting with function calling and stateful Plugins
{
    var myDescribedStatefulPlugin = new MyDescribedStatefulPlugin();
    var modelId = "llama3.2";
    var endpoint = new Uri("http://localhost:11434");
    var kernel = Kernel.CreateBuilder()
        .AddOllamaChatCompletion(modelId, endpoint)
        .Build();

    kernel.Plugins.AddFromObject(myDescribedStatefulPlugin);
    
    var service = kernel.GetRequiredService<IChatCompletionService>();

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

examples.Add(async () => // Embeddings Service
{
    var embeddingModelId = "mxbai-embed-large";
    var endpoint = new Uri("http://localhost:11434");
    var embeddingService = new EmbeddingGeneratorBuilder<string, Embedding<float>>()
                    .Use(new OllamaApiClient(endpoint, embeddingModelId))
                    .AsTextEmbeddingGenerationService();

    // Create a vector store and a collection to store information
    var vectorStore = new InMemoryVectorStore();
    var collection = vectorStore.GetCollection<string, SelectedModel>("ExampleCollection");
    await collection.CreateCollectionIfNotExistsAsync();

    // Save some information to the memory
    List<(string Id, string Detail)> factList = [
        ("phi3", "Scientific content"),
        ("llama3.2", "General chat content"),
        ("gpt-4o-mini", "Other subjects content")
    ];

    foreach (var (Id, Detail) in factList)
    {
        await collection.UpsertAsync(new()
        {
            ModelId = Guid.NewGuid().ToString(),
            PromptingDomain = Id,
            Embedding = await embeddingService.GenerateEmbeddingAsync(Detail)
        });
    }

    var vectorStoreTextSearch = new VectorStoreTextSearch<SelectedModel>(collection, embeddingService);

    while (true)
    {
        Console.Write($"\n\nUser > ");
        var userPrompt = Console.ReadLine();
        if (string.IsNullOrEmpty(userPrompt)) { break; }
        Console.WriteLine();
        var search = await vectorStoreTextSearch.SearchAsync(userPrompt, new TextSearchOptions { Top = 1 });
        await foreach (var item in search.Results)
        {
            Console.WriteLine($"Id: {item}");
        }
    }
});

examples.Add(async () => // Kernel simple chat model routing using model id
{
    var ollamaModelId = "llama3.2";
    var ollamaEndpoint = new Uri("http://localhost:11434");
    var fileModelId = "phi3";
    var fileModelPath = "E:\\repo\\huggingface\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
    var kernelBuilder = Kernel.CreateBuilder();

    kernelBuilder
            .AddOllamaChatCompletion(ollamaModelId, ollamaEndpoint)
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

examples.Add(async () => // Kernel simple chat model routing using service id
{
    var ollamaModelId = "llama3.2";
    var ollamaEndpoint = new Uri("http://localhost:11434");
    var fileModelId = "phi3";
    var fileModelPath = "E:\\repo\\huggingface\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
    var kernelBuilder = Kernel.CreateBuilder();

    kernelBuilder
        .Services
            .AddOllamaChatCompletion(
                serviceId: "ollama",
                modelId: ollamaModelId,
                endpoint: ollamaEndpoint)
            .AddOnnxRuntimeGenAIChatCompletion(
                serviceId: "onnx", 
                modelId: fileModelId, 
                modelPath: fileModelPath);
    // Last service will be the default service

    var kernel = kernelBuilder.Build();

    kernel.PromptRenderFilters.Add(new SelectedModelRenderFilter());

    while (true)
    {
        Console.Write($"\n\nType one of the available models > ");
        var selectedService = Console.ReadLine();
        Console.WriteLine();

        if (string.IsNullOrEmpty(selectedService)) { break; }

        var settings = new PromptExecutionSettings { ServiceId = selectedService };
        await foreach (var token in kernel.InvokePromptStreamingAsync("Answer in a small sentence, why does Jupiter have storms?", new(settings)))
        {
            Console.Write(token);
        };
    }
});

examples.Add(async () => // Kernel AI routing
{
    var ollamaModelId = "llama3.2";
    var ollamaEndpoint = new Uri("http://localhost:11434");
    var fileModelId = "phi3";
    var fileModelPath = "E:\\repo\\huggingface\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
    var kernelBuilder = Kernel.CreateBuilder();
    var openAIModelId = "gpt-4o-mini";

    kernelBuilder
        .AddOllamaChatCompletion(ollamaModelId, ollamaEndpoint)
        .AddOnnxRuntimeGenAIChatCompletion(fileModelId, fileModelPath)
        .AddOpenAIChatCompletion(openAIModelId, config["OpenAI:ApiKey"]!);

    // Last service will be the default service

    StringBuilder systemPrompt = new("""
            You are a model router and you will decide on only one of the 
            the models below that matches the capability more in line with the user's question. 
            Answer only with the Id of the model as the first word and a reason for it following.

            | Model Id | Capability |
            |----------|------------|
            """);

    // Router instructions
    List<(string, string)> modelCapabilities = [
        (ollamaModelId, "General chat"),
        (fileModelId, "Scientific chat"),
        (openAIModelId, "Other subjects")
    ];
    foreach (var (modelId, capability) in modelCapabilities)
    {
        systemPrompt.Append($"\n| {modelId} | {capability} |");
    }

    async Task<string> GetModelForPromptAsync(string prompt, string systemPrompt)
    {
        var routerService = new OpenAIChatCompletionService(
            modelId: "gpt-4o-mini",
            apiKey: config["OpenAI:ApiKey"]!);

        ChatHistory chatHistory = [
            new ChatMessageContent(AuthorRole.System, systemPrompt),
            new ChatMessageContent(AuthorRole.User, prompt)
        ];

        var selectionAndReason = (await routerService.GetChatMessageContentAsync(chatHistory)).ToString();
        var selectedModel = selectionAndReason[..selectionAndReason.IndexOf(' ')];

        Console.WriteLine($"\nSelected model by AI router: {selectionAndReason}\n");

        return selectedModel.ToString();
    }

    var kernel = kernelBuilder.Build();

    kernel.PromptRenderFilters.Add(new SelectedModelRenderFilter());

    var modelNames = string.Join(", ", kernel.GetAllServices<IChatCompletionService>().Select(s => s.GetModelId()));

    while (true)
    {
        Console.Write($"\n\nUser > ");
        var userPrompt = Console.ReadLine();
        if (string.IsNullOrEmpty(userPrompt)) { break; }
        Console.WriteLine();

        var selectedModel = await GetModelForPromptAsync(userPrompt, systemPrompt.ToString());

        var settings = new PromptExecutionSettings { ModelId = selectedModel };
        await foreach (var token in kernel.InvokePromptStreamingAsync(userPrompt, new(settings)))
        {
            Console.Write(token);
        };
    }
});

examples.Add(async () => // Kernel Embeddings Routing
{
    var ollamaModelId = "llama3.2";
    var ollamaEndpoint = new Uri("http://localhost:11434");
    var fileModelId = "phi3";
    var fileModelPath = "E:\\repo\\huggingface\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32";
    var kernelBuilder = Kernel.CreateBuilder();
    var openAIModelId = "gpt-4o-mini";

    kernelBuilder
        .AddOllamaChatCompletion(ollamaModelId, ollamaEndpoint)
        .AddOnnxRuntimeGenAIChatCompletion(fileModelId, fileModelPath)
        .AddOpenAIChatCompletion(openAIModelId, config["OpenAI:ApiKey"]!);
    // Last service will be the default service

    var embeddingModelId = "mxbai-embed-large";
    var endpoint = new Uri("http://localhost:11434");
    var embeddingService = new EmbeddingGeneratorBuilder<string, Embedding<float>>()
                    .Use(new OllamaApiClient(endpoint, embeddingModelId))
                    .AsTextEmbeddingGenerationService();
    
    // Create a vector store and a collection to store information
    var vectorStore = new InMemoryVectorStore();
    var collection = vectorStore.GetCollection<string, SelectedModel>("ExampleCollection");
    await collection.CreateCollectionIfNotExistsAsync();

    // Save some information to the memory
    List<(string Id, string Detail)> factList = [
        (fileModelId, "Scientific content"),
        (ollamaModelId, "General chat content"),
        (openAIModelId, "Other subjects content")
    ];

    foreach (var (Id, Detail) in factList)
    {
        await collection.UpsertAsync(new()
        {
            ModelId = Id,
            PromptingDomain = Detail,
            Embedding = await embeddingService.GenerateEmbeddingAsync(Detail)
        });
    }

    async Task<string> GetModelForPromptAsync(string prompt)
    {
        var searchVector = await embeddingService.GenerateEmbeddingAsync(prompt);
        var search = (await collection.VectorizedSearchAsync(searchVector, new() { Top = 1 }));
        string? selectedModel = null;
        string? selectedModelPromptingDomain = null;
        await foreach (var item in search.Results)
        {
            selectedModel = item.Record.ModelId;
            selectedModelPromptingDomain = item.Record.PromptingDomain;
        }
   
        Console.WriteLine($"\nSelected model by AI router: {selectedModel} {selectedModelPromptingDomain}\n");

        return selectedModel ?? "default";
    }

    var kernel = kernelBuilder.Build();

    kernel.PromptRenderFilters.Add(new SelectedModelRenderFilter());

    var modelNames = string.Join(", ", kernel.GetAllServices<IChatCompletionService>().Select(s => s.GetModelId()));

    while (true)
    {
        Console.Write($"\n\nUser > ");
        var userPrompt = Console.ReadLine();
        if (string.IsNullOrEmpty(userPrompt)) { break; }
        Console.WriteLine();

        var selectedModel = await GetModelForPromptAsync(userPrompt);

        var settings = new PromptExecutionSettings { ModelId = selectedModel };
        await foreach (var token in kernel.InvokePromptStreamingAsync(userPrompt, new(settings)))
        {
            Console.Write(token);
        };
    }
});

examples.Add(async () => // Multi modalities text -> audio services
{
    var chatService = new Azure.AI.Inference.ChatCompletionsClient(
        endpoint: new Uri(config["AzureAIInference:Endpoint"]!),
        credential: new Azure.AzureKeyCredential(config["AzureAIInference:ApiKey"]!))
    .AsChatClient("phi3")
    .AsChatCompletionService();

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
    string folderPath = @"C:\Users\rbarreto\OneDrive - Microsoft\Documents\Sound Recordings";

    DirectoryInfo directoryInfo = new(folderPath);

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
    string folderPath = @"C:\Users\rbarreto\OneDrive - Microsoft\Documents\Sound Recordings";

    DirectoryInfo directoryInfo = new(folderPath);

    // Get the most recent file created in the directory
    FileInfo? mostRecentFile = directoryInfo.GetFiles()
                                           .OrderByDescending(f => f.CreationTime)
                                           .FirstOrDefault();

    Console.WriteLine($"Most recent recording: {mostRecentFile!.FullName}");

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

examples.Add(async () => // Multi modalities image -> text -> audio)
{
    string folderPath = @"C:\Users\rbarreto\OneDrive - Microsoft\Pictures\Camera Roll";
    string huggingFaceModel = "Salesforce/blip-image-captioning-large";
    string openAIModel = "gpt-4o-mini";

    DirectoryInfo directoryInfo = new(folderPath);

    // Get the most recent image created in the directory
    FileInfo? mostRecentFile = directoryInfo.GetFiles("*.jpg")
                                           .OrderByDescending(f => f.CreationTime)
                                           .FirstOrDefault();

    Console.WriteLine($"Most recent photo: {mostRecentFile!.FullName}");

    var imageContent = new Microsoft.SemanticKernel.ImageContent(File.ReadAllBytes(mostRecentFile!.FullName), "image/jpeg");

    var huggingFaceImageToTextService = new HuggingFaceImageToTextService(huggingFaceModel, apiKey: config["HuggingFace:ApiKey"]!);

    var openAiChatService = new OpenAIChatCompletionService(modelId: openAIModel, apiKey: config["OpenAI:ApiKey"]!);

    Console.WriteLine("\nGenerating text from image (Hugging face) ...\n");
    var huggingFaceImageText = (await huggingFaceImageToTextService.GetTextContentsAsync(imageContent))[0];
    Console.WriteLine($"Hugging Face Text Generated:\n{huggingFaceImageText}\n");

    Console.WriteLine("\nGenerating text from image (OpenAI)...\n");
    ChatMessageContent chatMessageContent = new(AuthorRole.User, [
        new Microsoft.SemanticKernel.TextContent("Describe the image"),
        imageContent
    ]);
    var openAIImageText = await openAiChatService.GetChatMessageContentAsync([chatMessageContent]);
    Console.WriteLine($"OpenAI Text Generated:\n{openAIImageText}\n");

    Console.WriteLine("\nGenerating audio from text ...\n");
    var textToAudioService = new OpenAITextToAudioService(
        modelId: "tts-1",
        apiKey: config["OpenAI:ApiKey"]!);

    var huggingFaceGeneratedAudio = (await textToAudioService.GetAudioContentsAsync(huggingFaceImageText.ToString()))[0];
    var file = Path.Combine(Directory.GetCurrentDirectory(), "image-description-output.mp3");
    await File.WriteAllBytesAsync(file, huggingFaceGeneratedAudio.Data!.Value);

    var openAIGeneratedAudio = (await textToAudioService.GetAudioContentsAsync(openAIImageText.ToString()))[0];
    var file2 = Path.Combine(Directory.GetCurrentDirectory(), "image-description-output-2.mp3");
    await File.WriteAllBytesAsync(file2, openAIGeneratedAudio.Data!.Value);

    Console.WriteLine($"\nHuggingFace Audio Description. Ctrl + Click to listen: \n{new Uri(file).AbsoluteUri}");

    Console.WriteLine($"\nOpenAI Audio Description. Ctrl + Click to listen: \n{new Uri(file2).AbsoluteUri}");
});

examples.Add(async () => // Open Telemetry Aspire Dashboard 
{
    // docker run --rm -it -d -p 18888:18888 -p 4317:18889 --name aspire-dashboard mcr.microsoft.com/dotnet/aspire-dashboard:9.0

    var builder = Kernel.CreateBuilder();

    var oTelExporterEndpoint = "http://localhost:4317";

    var resourceBuilder = ResourceBuilder
        .CreateDefault()
        .AddService("TelemetryAspireDashboardQuickstart");

    // Enable model diagnostics with sensitive data.
    AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnosticsSensitive", true);

    using var traceProvider = Sdk.CreateTracerProviderBuilder()
        .SetResourceBuilder(resourceBuilder)
        .AddSource("Microsoft.SemanticKernel*")
        .AddOtlpExporter(options => options.Endpoint = new Uri(oTelExporterEndpoint))
        .Build();

    var meterProvider = Sdk.CreateMeterProviderBuilder()
        .AddMeter("Microsoft.SemanticKernel*");

    /*
        .SetResourceBuilder(resourceBuilder)
        .AddMeter("Microsoft.SemanticKernel*")
        .AddOtlpExporter(options => options.Endpoint = new Uri(oTelExporterEndpoint))
        .Build();*/

    using var loggerFactory = LoggerFactory.Create(builder =>
    {
        // Add OpenTelemetry as a logging provider
        builder.AddOpenTelemetry(options =>
        {
            options.SetResourceBuilder(resourceBuilder);
            options.AddOtlpExporter(options => options.Endpoint = new Uri(oTelExporterEndpoint));
            // Format log messages. This is default to false.
            options.IncludeFormattedMessage = true;
            options.IncludeScopes = true;
        });
        builder.SetMinimumLevel(LogLevel.Information);
    });

    builder.Services.AddSingleton(loggerFactory);

    builder.AddOpenAIChatCompletion(
        modelId: "gpt-4o-mini",
        apiKey: config["OpenAI:ApiKey"]!);

    Kernel kernel = builder.Build();

    while (true)
    {
        Console.Write("User > ");
        var userPrompt = Console.ReadLine();
        if (string.IsNullOrEmpty(userPrompt)) { break; }

        Console.Write("\nAssistant > ");
        await foreach (var token in kernel.InvokePromptStreamingAsync(userPrompt))
        {
            Console.Write(token);
        }

        Console.WriteLine("\n");
    }
});

// Dependency Injection

// Plugins within Plugins

// Kernel within Plugins

// Auto function filtering

// Prompt injection, PII filtering

// Function result filtering

// Open API plugins

#endregion Examples

await examples[^6](); // Run the last example

Console.WriteLine("\n\n\n");

#region Plugins

public class MyStatelessPlugin
{
    [KernelFunction]
    public static string GetTime()
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
        return this._counter++;
    }
}

public class MyDescribedStatefulPlugin
{
    private int _counter = 0;

    [KernelFunction, Description("Current count number")]
    [return: Description("The current count number")]
    public int GetCounter()
    {
        return this._counter++;
    }
}

#endregion

#region Filters

public class SelectedModelRenderFilter : IPromptRenderFilter
{
    public async Task OnPromptRenderAsync(PromptRenderContext context, Func<PromptRenderContext, Task> next)
    {
        Console.WriteLine($"Settings service id: {context.Arguments.ExecutionSettings?.FirstOrDefault().Value.ServiceId ?? "--"}");
        Console.WriteLine($"Settings model id: {context.Arguments.ExecutionSettings?.FirstOrDefault().Value.ModelId ?? "--"}");

        await next(context);
    }
}
#endregion

#region Embedding Models

internal sealed class SelectedModel
{
    [VectorStoreRecordKey]
    [TextSearchResultName]
    public string ModelId { get; set; } = string.Empty;

    [VectorStoreRecordData]
    [TextSearchResultValue]
    public string PromptingDomain { get; set; } = string.Empty;

    [VectorStoreRecordVector(Dimensions: 512)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}

#endregion

#region Debugging
public class CustomHttpClientHandler : HttpClientHandler
{
    protected async override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        // Console.WriteLine($"Request: {request.RequestUri}");
        // var requestBody = await request.Content!.ReadAsStringAsync();
        // Console.WriteLine($"Request Body: {requestBody}");
        return await base.SendAsync(request, cancellationToken);
    }
}
#endregion