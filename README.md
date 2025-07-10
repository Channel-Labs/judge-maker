<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">JudgeMaker</h3>

  <p align="center">
Optimize your LLM-as-a-judge prompt through reinforcement learning on synthetic data
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

### What is JudgeMaker?

JudgeMaker is the first solution to fine-tune models for creating optimal LLM-as-a-judge prompts that align with your users' preferences, all without requiring any user data. It bridges the gap between prompt engineering and fine-tuning by generating judge prompts that are both effective and transparent.

### The Problem with Current LLM Evaluation

LLM-as-a-judge has become the standard for evaluating language models at scale, but current approaches have significant limitations:

**Prompt Engineering Approach:**
- Relies on manual human effort to craft effective prompts
- Difficult to systematically optimize for specific use cases
- Limited scalability and consistency

**Fine-Tuning Approach:**
- Requires extensive human preference data
- Creates black-box models with no transparency into judging criteria
- Expensive and time-consuming to implement

### Our Solution

JudgeMaker combines the advantages of both approaches. Instead of fine-tuning a model to judge responses directly, we fine-tune a model to generate judge prompts that align with human preferences. This maintains full transparency while achieving superior performance.

#### Key Innovations

**🎭 Synthetic Conversations Backed by Personas**
We generate high-quality synthetic conversations by first creating diverse, detailed personas representing your users. Each persona includes specific characteristics, scenarios, and use cases that allow an LLM to predict preferences accurately. This approach builds on established libraries like [DiaSynth](https://github.com/ntuspeechlab/DiaSynth) and our own [synthetic conversation generation](https://github.com/Channel-Labs/synthetic-conversation-generation) library.

**🔄 Reinforcement Learning on Verifiable Rewards**
Our breakthrough enables true reinforcement learning for preference alignment. By giving the reward model full access to persona information while keeping the prompt-generating model blind to these details, we create verifiable rewards at scale. The model learns to generate prompts that help LLMs align with synthetic human preferences without ever seeing the underlying persona data.

### What does this repo do?

**Create a Fine-Tuned Model**
Launches RL fine-tuning jobs that train models to generate judge prompts matching your users' preferences

**Evaluate  Model Performance**
Tests models against multi-turn conversations from Chatbot Arena to measure how well generated prompts enable LLMs to predict human preferences.

---

*For detailed technical information about our approach, methodology, and experimental results, please refer to our [technical white paper](XXXXX).*

<!-- GETTING STARTED -->
## Getting Started

Here's how to set up the Synthetic Conversation Generation toolkit.

### Prerequisites

* Python 3.9+
* pip (Python package manager)
* API keys for your chosen LLM provider 
  - [OpenAI](https://platform.openai.com/docs/overview)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/channel-labs/synthetic-conversation-generation.git
   ```
2. Install Python dependencies
   ```sh
   pip install -r requirements.txt
   ```
3. Install the package in development mode
   ```sh
   pip install -e .
   ```
4. Set up environment variables for your API keys
   ```sh
   export OPENAI_API_KEY='your_openai_api_key'
   ```

<!-- USAGE EXAMPLES -->
## Usage

The toolkit provides two main functionalities:

### 1. User Persona Generation

Generate a diverse set of realistic user personas tailored to your AI assistant. These personas capture a range of backgrounds, goals, and interaction styles, enabling robust and representative conversation simulations.

**Usage:**

```sh
python src/synthetic_conversation_generation/persona_generator.py \
  --assistant-name <ASSISTANT_NAME> \
  --assistant-description <ASSISTANT_DESCRIPTION> \
  --num-personas <NUM_PERSONAS> \
  --output-path <OUTPUT_PATH> \
  --model-provider <MODEL_PROVIDER> \
  --model-id <MODEL_ID>
```

**Arguments:**

- `--assistant-name`: Name of your AI assistant (e.g., "TaxBot").
- `--assistant-description`: Brief description of your assistant's purpose and capabilities.
- `--num-personas`: Number of user personas to generate (default: 5).
- `--output-path`: Path to save the generated personas (YAML format).
- `--model-provider`: LLM provider to use (`openai` or `anthropic`, default: `openai`).
- `--model-id`: Model ID for persona generation (default: `gpt-4.1`).

**Example:**

```sh
python src/synthetic_conversation_generation/persona_generator.py \
  --assistant-name "Fashionable Fran" \
  --assistant-description "A personal stylist recommending outfits from your wardrobe." \
  --num-personas 3 \
  --output-path examples/conversation_characters/fashion_advisor.yaml
```

After generation, review and optionally edit the personas in the output YAML file to ensure they fit your use case.

### 2. Conversation Simulation

Generate realistic conversations between synthetic users and your AI assistant. The system intelligently creates user messages that match the personas from step 1, calls your AI endpoint for responses, and determines natural conversation endpoints. Unlike typical synthetic data generators, this system dynamically evaluates whether a conversation should continue or end naturally. After each turn, it determines if the user's needs have been met or if the conversation has reached a logical conclusion, creating more realistic dialogue patterns.

```sh
python src/synthetic_conversation_generation/conversation_generator.py \
  --conversation-characters-path <CONVERSATION_CHARACTERS_PATH> \
  --inference-endpoint-path <INFERENCE_ENDPOINT_PATH> \
  --max-conversation-turns <MAX_CONVERSATION_TURNS> \
  --output-path <OUTPUT_PATH> \
  --model-provider <MODEL_PROVIDER> \
  --model-id <MODEL_ID>
```

**Arguments:**

- `--conversation-characters-path`: Path to the YAML file containing the assistant details and user personas (output from persona_generator).
- `--inference-endpoint-path`: Path to a YAML file specifying how to call your AI assistant via HTTP.
- `--max-conversation-turns`: Maximum number of turns a conversation can have (default: 10).
- `--output-path`: Path to save the generated conversations (JSONL format).
- `--model-provider`: LLM provider to use for generating user messages (`openai` or `anthropic`, default: `openai`).
- `--model-id`: Model ID for generating user messages (default: `gpt-4.1`).

**Example:**

```sh
python src/synthetic_conversation_generation/conversation_generator.py \
  --conversation-characters-path examples/conversation_characters/fashion_advisor.yaml \
  --inference-endpoint-path examples/endpoint/openai_chat_completion.yaml \
  --max-conversation-turns 10 \
  --output-path examples/conversations/fashion_advisor_conversations.jsonl
```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Otherwise, feel free to start a discussion or open an issue here on GitHub, and we'll review shortly.

Don't forget to give the project a star! Thanks again!

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Create by [Channel Labs](https://channellabs.ai/)

Interested in understand and improving your AI's behavior even further? Contact scott@channellabs.ai for any inquiries.
