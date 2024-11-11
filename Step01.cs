﻿// Step 1, Kernel
// Chat Service
// Blocking Chat Completion Call

using Microsoft.SemanticKernel;

var modelId = "llama3.2";
var endpoint = new Uri("http://localhost:11434");

var kernel = Kernel.CreateBuilder()
    .AddOllamaChatCompletion(modelId, endpoint)
    .Build();

var response = await kernel.InvokePromptAsync("Hello, how are you?");

Console.WriteLine(response);

