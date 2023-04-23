using Azure;
using Azure.AI.OpenAI;
using Microsoft.ANN.SPTAGManaged;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.DirectoryServices;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DebuggerHelp
{
    public class SearchResult
    {
        public static string baseUrl = "https://learn.microsoft.com/windows-hardware/drivers/debugger";
        public Uri Uri => new Uri($"{baseUrl}/{System.IO.Path.GetFileNameWithoutExtension(Name)}");
        public string Name { get; set; }
        public string Title
        {
            get
            {
                var unescaped = System.IO.Path.GetFileNameWithoutExtension(Name).Replace('-', ' ').Trim();
                // replace multiple spaces with one space
                var singleSpaced = new Regex(@"\s+").Replace(unescaped, " ");
                return singleSpaced;
            }
        }
    }
    /// <summary>
    /// Interaction logic for HelpPage.xaml
    /// </summary>
    public partial class HelpPage : Page, INotifyPropertyChanged
    {
        public ObservableCollection<SearchResult> SearchResults { get; private set; } = new();
        AnnIndex m_index;
        OpenAIClient m_openAIClient;

        public static readonly DependencyProperty AzureOpenAIEndpointProperty = DependencyProperty.Register("AzureOpenAIEndpoint", typeof(string), typeof(HelpPage), new PropertyMetadata(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")));
        public static readonly DependencyProperty AzureOpenAIKeyProperty = DependencyProperty.Register("AzureOpenAIKey", typeof(string), typeof(HelpPage), new PropertyMetadata(Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY")));
        public static readonly DependencyProperty IndexPathProperty = DependencyProperty.Register("IndexPath", typeof(string), typeof(HelpPage), new PropertyMetadata());
        public static readonly DependencyProperty EmbeddingDeploymentNameProperty = DependencyProperty.Register("EmbeddingDeploymentName", typeof(string), typeof(HelpPage), new PropertyMetadata(Environment.GetEnvironmentVariable("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")));

        public string AzureOpenAIEndpoint
        {
            get { return (string)GetValue(AzureOpenAIEndpointProperty); }
            set { SetValue(AzureOpenAIEndpointProperty, value); }
        }

        public string AzureOpenAIKey
        {
            get { return (string)GetValue(AzureOpenAIKeyProperty); }
            set { SetValue(AzureOpenAIKeyProperty, value); }
        }
        
        public string IndexPath
        {
            get { return (string)GetValue(IndexPathProperty); }
            set { SetValue(IndexPathProperty, value); }
        }

        public string EmbeddingDeploymentName
        {
            get { return (string)GetValue(EmbeddingDeploymentNameProperty); }
            set { SetValue(EmbeddingDeploymentNameProperty, value); }
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        public HelpPage()
        {
            InitializeComponent();
            results.DataContext = this;
        }

        public async void Page_Loaded(object sender, RoutedEventArgs e)
        {
            m_index = AnnIndex.Load(IndexPath);
            m_openAIClient = new OpenAIClient(new Uri(AzureOpenAIEndpoint), new AzureKeyCredential(AzureOpenAIKey));
            await webview.EnsureCoreWebView2Async();
        }

        private byte[] GetEmbeddingBytes(IReadOnlyList<float> embedding)
        {
            // get the bytes from embedding
            return embedding.SelectMany(e => BitConverter.GetBytes(e)).ToArray();
        }

        private async Task<BasicResult[]> Search(OpenAIClient client, AnnIndex index, string query)
        {
            var queryEmbedding = (await client.GetEmbeddingsAsync(EmbeddingDeploymentName, new EmbeddingsOptions(query)));
            var queryBytes = GetEmbeddingBytes(queryEmbedding.Value.Data[0].Embedding);
            return index.SearchWithMetaData(queryBytes, 15);
        }

        private async Task Submit()
        {
            var queryText = query.Text;
            var results = await Search(m_openAIClient, m_index, queryText);
            SearchResults.Clear();
            HashSet<string> titles = new HashSet<string>();
            foreach (var result in results)
            {
                var metadata = Encoding.ASCII.GetString(result.Meta).Split('|');
                if (titles.Contains(metadata[0])) continue;
                SearchResults.Add(new SearchResult { Name = metadata[0] });
                titles.Add(metadata[0]);
            }
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(SearchResults)));
        }

        private async void query_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await Submit();
                e.Handled = true;
            }
        }

        private void TextBlock_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            var result = (sender as TextBlock)!.DataContext as SearchResult;

            webview.Source = result!.Uri;
        }

        private void webview_CoreWebView2InitializationCompleted(object sender, Microsoft.Web.WebView2.Core.CoreWebView2InitializationCompletedEventArgs e)
        {
            webview.NavigateToString("<!DOCTYPE html>" +
                "<html lang=\"en\" xmlns=\"http://www.w3.org/1999/xhtml\">" +
                "<head>" +
                    "<meta charset=\"utf-8\" />" +
                    "<title>Landing page</title>" +
                "</head>" +
                "<body>" +
                    "<p>Type something in the text box to search for something</p>" +
                "</body>" +
                "</html>");
            query.Focus();
        }
    }
}
