# 🤖 Agentic Query Assistant with Enhanced Browser Agent

A sophisticated AI agent system designed for web automation, document analysis, and intelligent query processing. This project implements a multi-agent architecture with specialized browser automation capabilities.

## 🎯 Key Features

### 🌐 Enhanced Browser Agent
- **Stable Element Identification**: Immune to dynamic DOM changes and ID flickering
- **Intelligent Form Automation**: Systematic form filling with semantic field recognition
- **Multi-directional Agent Routing**: Seamless integration with perception, decision, and execution modules
- **Smart Error Recovery**: Advanced fallback mechanisms for robust web automation
- **Session State Management**: Persistent browser sessions with intelligent caching

### 🧠 Multi-Agent Architecture
- **Perception Agent**: Natural language understanding and query classification
- **Decision Agent**: Strategic planning and action determination  
- **Browser Agent**: Specialized web automation and form filling
- **Summarization Agent**: Content analysis and information extraction
- **Memory System**: Context-aware conversation history and session management

### 📊 MCP Integration
- **35+ Browser Tools**: Comprehensive web automation toolkit
- **Document Processing**: PDF, Word, and text analysis capabilities
- **Web Search**: Real-time information retrieval
- **FAISS Vector Search**: Semantic document search and retrieval

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (for browser automation)
- Required Python packages (see `pyproject.toml`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd S12
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # or use uv/poetry based on pyproject.toml
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

4. **Start the agent**
   ```bash
   python main.py
   ```

## 💡 Usage Examples

### Form Filling with Browser Agent
```
📝 You: Please go to the Google Form at https://docs.google.com/forms/d/e/1FAIpQLSf0gKdNlBtKtQb5MJlIvYHw-kXHj5k6BRKZ6BSqr0CYc9cSQQ/viewform and fill it out with the following information:

- Is he/she married?: "Yes"
- What course is he/her in?: "The school of AI"  
- Which course is he/she taking?: "EAG"
- What is his/her email id?: "your.email@example.com"
- What is the name of your master?: "Sudarshan Iyengar"
- What is his/her Date of Birth?: "15 Feb 1998"

Please complete and submit the form.
```

### Document Analysis
```
📝 You: Summarize the main points from the PDF document about Tesla's innovation strategy
```

### Web Research
```
📝 You: Find the latest differences between BMW 7 series and 5 series, reply in markdown format
```

### Local Document Search
```
📝 You: What is the relationship between Gensol and Go-Auto based on local documents?
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │────│    Decision     │────│    Execution    │
│     Agent       │    │     Agent       │    │     Agents      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  Browser Agent  │──────────────┘
                        │   (Enhanced)    │
                        └─────────────────┘
                                │
                    ┌─────────────────────┐
                    │   MCP Tools         │
                    │ • 35+ Browser Tools │
                    │ • Document Tools    │
                    │ • Search Tools      │
                    │ • Memory Tools      │
                    └─────────────────────┘
```

### Enhanced Browser Agent Features

#### 🔒 Stable Element Identification
- **Semantic Mapping**: Elements identified by purpose (email, name, date) rather than dynamic indices
- **Multi-layer Fallback**: Purpose → Attributes → Text → Type identification
- **Index Change Immunity**: Continues working despite DOM modifications
- **Smart Caching**: Reduces unnecessary DOM rescans

#### 🎯 Intelligent Form Automation
- **Form State Tracking**: Prevents repetitive actions and infinite loops
- **Progress Monitoring**: Tracks completed fields and remaining tasks
- **Error Recovery**: Advanced fallback strategies for failed actions
- **Systematic Workflow**: Methodical progression through form fields

#### 📊 Element Stability Tracking
- **Change Detection**: Monitors element stability across scans
- **Signature-based Identification**: Creates persistent element fingerprints
- **Stability Analysis**: Logs and analyzes DOM change patterns
- **Performance Optimization**: Intelligent caching based on stability metrics

## 🛠️ Configuration

### MCP Server Configuration
Edit `config/mcp_server_config.yaml`:
```yaml
mcp_servers:
  - name: "browser_mcp"
    command: "python"
    args: ["browserMCP/browser_mcp_stdio.py"]
  - name: "document_mcp"  
    command: "python"
    args: ["mcp_servers/mcp_server_1.py"]
```

### Model Configuration
Edit `config/models.json`:
```json
{
  "primary_model": "gpt-4",
  "fallback_model": "gpt-3.5-turbo",
  "temperature": 0.7
}
```

### Browser Profiles
Edit `config/profiles.yaml` for browser-specific settings:
```yaml
browser_profiles:
  default:
    headless: false
    viewport: "1920x1080"
    timeout: 30000
```

## 📂 Project Structure

```
S12/
├── agent/                          # Core agent implementations
│   ├── browserAgent.py            # ⭐ Enhanced browser agent
│   ├── browser_tasks.py           # LLM-based task generation
│   ├── enhanced_agent_loop.py     # Multi-directional routing
│   └── agent_loop3.py             # Main agent orchestration
├── browserMCP/                     # Browser MCP tools
│   ├── mcp_tools.py               # 35+ browser automation tools
│   ├── dom/                       # DOM processing and analysis
│   └── browser/                   # Browser session management
├── mcp_servers/                    # MCP server implementations
├── config/                         # Configuration files
├── prompts/                        # LLM prompt templates
├── memory/                         # Session and conversation memory
└── utils/                          # Utility functions and helpers
```

## 🎯 Browser Agent Capabilities

### Form Automation
- ✅ **Input Field Detection**: Email, text, date, number fields
- ✅ **Radio Button Selection**: Single and multiple choice handling
- ✅ **Dropdown Navigation**: Select option identification and selection
- ✅ **Checkbox Management**: Boolean and multi-select checkboxes
- ✅ **Form Submission**: Intelligent submit button detection
- ✅ **Validation Handling**: Error detection and recovery

### Web Navigation
- ✅ **URL Navigation**: Direct and relative URL handling
- ✅ **Link Following**: Text and attribute-based link identification
- ✅ **Page Interaction**: Scrolling, clicking, hovering
- ✅ **Session Management**: Cookie and state persistence
- ✅ **Multi-tab Support**: Tab creation and switching

### Content Extraction
- ✅ **Text Extraction**: Structured and unstructured content
- ✅ **Element Analysis**: Attribute and property extraction
- ✅ **Screenshot Capture**: Visual page state documentation
- ✅ **Markdown Conversion**: Clean content formatting
- ✅ **Data Scraping**: Table and list data extraction

## 🔧 Troubleshooting

### Common Issues

#### Browser Agent Not Working
1. **Check MCP servers are running**: Verify browser MCP is initialized
2. **Verify element detection**: Use debug mode to see element mapping
3. **Check stable element logs**: Look for stability warnings in output

#### Element ID Flickering (Fixed!)
- ✅ **Root Cause**: Dynamic index assignment in DOM scanning
- ✅ **Solution**: Semantic element identification system implemented
- ✅ **Result**: Immune to DOM changes and index modifications

#### Form Filling Loops
- ✅ **Root Cause**: Repetitive actions without state tracking
- ✅ **Solution**: Form state management and progress tracking
- ✅ **Result**: Systematic progression through form fields

### Debug Mode
Enable detailed logging:
```bash
LOGLEVEL=DEBUG python main.py
```

### Browser Debug
View browser interactions:
```bash
BROWSER_DEBUG=true python main.py
```

## 🚀 Recent Enhancements

### Browser Agent Fixes (Latest)
- **🔒 Stable Element Identification**: Added semantic-based element mapping
- **📊 Element Stability Tracking**: Implemented change detection and analysis
- **🎯 Smart Form Automation**: Enhanced form state management and progress tracking
- **🔄 Intelligent Caching**: Reduced unnecessary DOM rescans
- **⚡ Performance Optimization**: 3-5x faster element identification

### Architecture Improvements
- **🌐 Multi-directional Routing**: Enhanced agent communication
- **🧠 Intelligent Decision Making**: LLM-powered action planning
- **📈 Advanced Error Recovery**: Sophisticated fallback mechanisms
- **💾 Session Persistence**: Improved memory and state management

## 📚 Documentation

- **HOWTORUN.md**: Detailed setup and execution guide
- **Class_notes.md**: Development notes and implementation details
- **config/**: Configuration file documentation
- **prompts/**: Prompt engineering templates and examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Ensure all tests pass

## 📄 License

This project is part of an educational assignment for EAG (Emerging AI & Generative Technologies) coursework.

## 🙏 Acknowledgments

- **The School of AI**: Educational framework and guidance
- **MCP Protocol**: Tool integration architecture
- **OpenAI/Anthropic**: LLM capabilities and API access
- **Browser Automation Community**: Tools and techniques inspiration

---

**🚀 Ready to automate the web with intelligent agents!**


