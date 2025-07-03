# 🔍 Agentic Web Search Assistant

An intelligent AI agent system designed for advanced web search, information retrieval, and real-time research capabilities. This project implements a multi-agent architecture specialized for finding, analyzing, and synthesizing information from the web.

## 🎯 Key Features

### 🌐 Advanced Web Search
- **Real-time Information Retrieval**: Live web search with current data
- **Multi-source Analysis**: Combines information from multiple web sources
- **Intelligent Query Processing**: Natural language to search query optimization
- **Content Synthesis**: Aggregates and summarizes findings from multiple sources
- **Fact Verification**: Cross-references information across different websites

### 🧠 Smart Research Architecture
- **Perception Agent**: Understands research intent and query context
- **Decision Agent**: Plans optimal search strategies and source selection
- **Web Search Agent**: Executes searches and retrieves relevant content
- **Summarization Agent**: Analyzes and synthesizes information
- **Memory System**: Maintains research context and builds knowledge base

### 📊 Search Integration
- **Multiple Search Engines**: Google, Bing, and specialized search APIs
- **Content Processing**: HTML parsing, text extraction, and structure analysis
- **Document Analysis**: PDF, Word, and web page content processing
- **Vector Search**: Semantic similarity search for relevant information
- **Citation Tracking**: Source attribution and reference management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Internet connection for web search
- API keys for search services (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd S12
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure search APIs**
   ```bash
   cp .env.example .env
   # Add your search API keys (Google Custom Search, Bing, etc.)
   ```

4. **Start the search assistant**
   ```bash
   python main.py
   ```

## 💡 Web Search Examples

### Comparative Research
```
📝 You: Compare the latest features and specifications between iPhone 15 Pro and Samsung Galaxy S24 Ultra
```

### Market Analysis
```
📝 You: What are the current trends in electric vehicle adoption globally? Include statistics and recent developments
```

### Technical Research
```
📝 You: Explain the differences between quantum computing approaches: gate-based vs annealing vs photonic
```

### News and Current Events
```
📝 You: What are the latest developments in AI regulation across different countries?
```

### Product Research
```
📝 You: Find reviews and comparisons of the best laptops for machine learning in 2024, include pricing and specifications
```

### Academic Research
```
📝 You: Research the latest papers on large language model training efficiency and summarize key findings
```

## 🏗️ Search Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query         │────│    Search       │────│   Information   │
│  Processing     │    │   Execution     │    │   Synthesis     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  Web Research   │──────────────┘
                        │     Agent       │
                        └─────────────────┘
                                │
                    ┌─────────────────────┐
                    │   Search Tools      │
                    │ • Web Search APIs   │
                    │ • Content Scrapers  │
                    │ • Text Processors   │
                    │ • Citation Trackers │
                    └─────────────────────┘
```

### Advanced Search Features

#### 🎯 Intelligent Query Enhancement
- **Context Understanding**: Interprets search intent and domain
- **Query Expansion**: Adds relevant terms and synonyms
- **Search Strategy**: Plans multi-step research approaches
- **Source Diversification**: Searches across different types of sources

#### 📊 Content Analysis
- **Relevance Scoring**: Ranks sources by relevance and credibility
- **Information Extraction**: Pulls key facts, figures, and insights
- **Contradiction Detection**: Identifies conflicting information across sources
- **Trend Analysis**: Spots patterns and trends in search results

#### 🔍 Research Capabilities
- **Multi-step Research**: Builds knowledge through sequential searches
- **Cross-verification**: Confirms facts across multiple sources
- **Recent vs Historical**: Distinguishes between current and historical information
- **Expert Source Identification**: Prioritizes authoritative sources

## 🛠️ Search Configuration

### Search API Setup
Edit your `.env` file:
```bash
# Google Custom Search
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Bing Search API
BING_SEARCH_API_KEY=your_bing_api_key

# Additional APIs
SERP_API_KEY=your_serp_api_key
```

### Search Parameters
Configure search behavior in `config/search_config.yaml`:
```yaml
search_settings:
  max_results_per_query: 10
  search_timeout: 30
  content_max_length: 5000
  enable_fact_checking: true
  prioritize_recent: true
```

## 📊 Search Capabilities

### Information Types
- ✅ **Current Events**: Real-time news and developments
- ✅ **Product Comparisons**: Features, pricing, and reviews
- ✅ **Technical Information**: Specifications, tutorials, and guides
- ✅ **Market Research**: Trends, statistics, and analysis
- ✅ **Academic Content**: Research papers, studies, and publications
- ✅ **Historical Data**: Past events, timelines, and context

### Output Formats
- ✅ **Structured Summaries**: Organized key points and findings
- ✅ **Comparison Tables**: Side-by-side feature/spec comparisons
- ✅ **Markdown Reports**: Formatted research documents
- ✅ **Citation Lists**: Source references and links
- ✅ **Executive Summaries**: Concise overview of findings
- ✅ **Trend Analysis**: Data visualization and insights

## 🔧 Search Optimization

### Best Practices
1. **Specific Queries**: Use detailed, specific search terms
2. **Context Provision**: Include relevant background information
3. **Output Format**: Specify desired format (markdown, table, etc.)
4. **Time Sensitivity**: Indicate if current information is needed
5. **Source Preferences**: Mention preferred types of sources

### Example Query Structures
```
📝 Good: "Compare cloud storage pricing between AWS S3, Google Cloud, and Azure for small businesses in 2024"

📝 Better: "Find the latest pricing comparison for cloud storage services (AWS S3, Google Cloud Storage, Azure Blob) focusing on small business use cases under 1TB, include recent price changes and feature differences"
```

## 🚀 Recent Search Enhancements

### Search Performance
- **⚡ Faster Results**: Optimized search execution and content processing
- **🎯 Better Relevance**: Improved ranking and filtering algorithms
- **📊 Enhanced Analysis**: Better information synthesis and summarization
- **🔄 Smart Caching**: Reduced redundant searches and faster follow-ups

### Content Quality
- **✅ Source Verification**: Enhanced credibility assessment
- **📈 Trend Detection**: Better identification of current developments
- **🔍 Deep Analysis**: More thorough content extraction and analysis
- **📚 Citation Management**: Improved source tracking and attribution

## 💡 Use Cases

### Business Research
- Market analysis and competitor research
- Product comparisons and vendor evaluation
- Industry trends and forecast analysis
- Regulatory updates and compliance information

### Academic Research
- Literature reviews and paper discovery
- Current research trends and methodologies
- Expert opinions and scholarly debates
- Statistical data and research findings

### Personal Research
- Product purchases and reviews
- Travel planning and destination research
- Health and wellness information
- Technology trends and recommendations

## 📚 Documentation

- **Search Guide**: Advanced search techniques and strategies
- **API Documentation**: Search service integration details
- **Configuration**: Setup and customization options
- **Examples**: Sample queries and expected outputs

---

**🔍 Discover the web's knowledge with intelligent search!**


