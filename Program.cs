using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using OllamaSharp;
using System.ComponentModel;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

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

    var result = await kernel.InvokePromptAsync(myPrompt, arguments);
    Console.WriteLine(result);

    result = await kernel.InvokePromptAsync(myPrompt, arguments);
    Console.WriteLine(result);
});

#endregion Examples

await examples.Last()();

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