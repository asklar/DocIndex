using Azure;
using Azure.AI.OpenAI;
using Azure.Core;
using Azure.Identity;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.ANN.SPTAGManaged;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics.SymbolStore;

string? apikey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");
string? endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
string? embeddingDeploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME");
string localFolder = Environment.CurrentDirectory;
string projectName = "vector_index";
var maxTokensPerChunk = 4096;
var charactersPerToken = 2.5;
bool search = false;
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "-apiKey")
    {
        apikey = args[++i];
    }
    else if (args[i] == "-endpoint")
    {
        endpoint = args[++i];
    }
    else if (args[i] == "-folder")
    {
        localFolder = args[++i];
    }
    else if (args[i] == "-project")
    {
        projectName = args[++i];
    }
    else if (args[i] == "-tokensPerChunk")
    {
        maxTokensPerChunk = int.Parse(args[++i]);
    }
    else if (args[i] == "-charsPerToken")
    {
        charactersPerToken = double.Parse(args[++i]);
    }
    else if (args[i] == "-embeddingDeploymentName")
    {
        embeddingDeploymentName = args[++i];
    }
    else if (args[i] == "-search")
    {
        search = true;
    }
    else if (args[i] == "-?")
    {
        Console.WriteLine("DocIndex [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine();
        Console.WriteLine("  -apiKey <key>         - Azure OpenAI API key");
        Console.WriteLine("  -endpoint <endpoint>  - Azure OpenAI endpoint");
        Console.WriteLine("  -embeddingDeploymentName <name>  - Azure OpenAI embedding deployment name");
        Console.WriteLine("  -folder <folder>      - Folder to index");
        Console.WriteLine($"  -project <project>    - Project name (default: {projectName})");
        Console.WriteLine($"  -tokensPerChunk <n>   - Max number of tokens per chunk (default: {maxTokensPerChunk})");
        Console.WriteLine($"  -charsPerToken <n>    - Average number of characters per token (default: {charactersPerToken})");
        Console.WriteLine("  -?                    - This help");
        return;
    }
}

if (apikey == null) {
    Console.WriteLine("Missing API key. Please set the AZURE_OPENAI_KEY environment variable or specify -apiKey ...");
    return;
}
else if (endpoint == null)
{
    Console.WriteLine("Missing endpoint. Please set the AZURE_OPENAI_ENDPOINT environment variable or specify -endpoint ...");
    return;
}
else if (embeddingDeploymentName == null)
{
    Console.WriteLine("Missing embedding deployment name. Please set the AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable or specify -embeddingDeploymentName ...");
    return;
}

OpenAIClient client = new OpenAIClient(new Uri(endpoint!), new AzureKeyCredential(apikey!));

const int embeddingDimensions = 1536;
var folder = Path.Combine(localFolder, projectName);
Console.OutputEncoding = Encoding.UTF8;

if (search)
{
    await SearchFolderRepl(folder, localFolder, client);
}
else
{
    await IndexFolder(folder);
}
async void Repl(string folder)
{
    Console.WriteLine("1. Index");
    Console.WriteLine("2. Search");
    var opt = Console.Read();
    if ((char)opt == '1')
    {
        await IndexFolder(folder);
    }
    else
    {
        await SearchFolderRepl(folder, localFolder, client);
        return;
    }
    Console.WriteLine("Done");
    Console.ReadKey();
}

static async Task SearchFolderRepl(string folder, string localFolder, OpenAIClient client)
    {
        Console.ReadLine();
        var savedIdx = Path.Combine(folder, "index.sptag");
        var currentTime = DateTime.Now;
        var idx = AnnIndex.Load(savedIdx);
        var duration = DateTime.Now - currentTime;
        Console.WriteLine($"loaded index in {duration.TotalMilliseconds} ms\n");
        while (true)
        {
            var query = Console.ReadLine()!;
            var result = await Search(client, idx, query);
            Console.WriteLine("******************************************************************");
            foreach (var r in result)
            {
                var metadata = Encoding.ASCII.GetString(r!.Meta).Split('|');
                var filename = metadata[0];
                var startPos = int.Parse(metadata[1]);
                var distance = r.Dist;

                Console.WriteLine($"Result #{r.VID} - {filename} - distance={distance}");
                var text = File.ReadAllText(Path.Combine(localFolder, filename));
                var chunk = text.Substring(startPos, Math.Min(100, text.Length - startPos));
                Console.WriteLine($"...{chunk}");
            }
        }
    }

async Task IndexFolder(string folder)
{
    var index = new AnnIndex("BKT", "Float", embeddingDimensions);
    index.SetBuildParam("DistCalcMethod", "L2", "Index");
    var files = Directory.GetFiles(folder);
    for (int i  = 0; i < files.Length; i++) { 
        var file = files[i];
        var nChunks = await EmbedFileAsync(file, index);
        Console.WriteLine($"✅ {file} - {nChunks} chunks - {i}/{files.Length} ({(i*100.0/files.Length):0.#}%)");
    }
    var savedIdx = Path.Combine(folder, "index.sptag");
    index.Save(savedIdx);
}

async Task<int> EmbedFileAsync(string file, AnnIndex index)
{
    var textToEmbed = File.ReadAllText(file);

    var embeddings = new List<IReadOnlyList<float>>();
    var nChunks = (int)Math.Ceiling((double)textToEmbed.Length / (maxTokensPerChunk * charactersPerToken));
    var relativePath = Path.GetRelativePath(localFolder, file);
    var totalChunks = 0;
    for (int i = 0; i < nChunks; i++)
    {
        var start = (int) Math.Floor(i * charactersPerToken * maxTokensPerChunk);
        var end = (int)Math.Min((i + 1) * charactersPerToken * maxTokensPerChunk, textToEmbed.Length);
        var chunkText = textToEmbed.Substring(start, end - start);

        try
        {
            await EmbedIntoIndex(index, client, relativePath, chunkText, start, embeddingDeploymentName!);
            totalChunks++;
        }
        catch (Azure.RequestFailedException)
        {
            var middle = (start + end) / 2;
            var first = textToEmbed.Substring(start, middle - start);
            var second = textToEmbed.Substring(middle, end - middle);
            try
            {
                await EmbedIntoIndex(index, client, relativePath, first, start, embeddingDeploymentName!);
                totalChunks++;
                await EmbedIntoIndex(index, client, relativePath, second, middle, embeddingDeploymentName!);
                totalChunks++;
            }
            catch (Azure.RequestFailedException ex)
            {
                var x = ex.Message;
                Console.WriteLine($"Error while processing {file} - {x}");
            }
        }
    }
    return totalChunks;
    static async Task EmbedIntoIndex(AnnIndex index, OpenAIClient client, string relativePath, string chunkText, int start, string embeddingDeploymentName)
    {
        var embed = (await client.GetEmbeddingsAsync(embeddingDeploymentName, new EmbeddingsOptions(chunkText)));
        Debug.Assert(embed.Value.Data.Count == 1);

        for (int j = 0; j < embed.Value.Data.Count; j++)
        {
            var embedding = embed.Value.Data[j].Embedding;
            byte[] embeddingBytes = GetEmbeddingBytes(embedding);
            var metadata = $"{relativePath}|{start}";
            var b = index.AddWithMetaData(embeddingBytes, Encoding.ASCII.GetBytes(metadata), 1, false, false);
        }
    }
}



static byte[] GetEmbeddingBytes(IReadOnlyList<float> embedding)
{
    // get the bytes from embedding
    return embedding.SelectMany(e => BitConverter.GetBytes(e)).ToArray();
}

static async Task<BasicResult[]> Search(OpenAIClient client, AnnIndex index, string query)
{
    var queryEmbedding = (await client.GetEmbeddingsAsync("Text-Embedding-ADA-002", new EmbeddingsOptions(query)));
    var queryBytes = GetEmbeddingBytes(queryEmbedding.Value.Data[0].Embedding);
    return index.SearchWithMetaData(queryBytes, 15);
}