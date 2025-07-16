<!-- PROJECT LOGO -->
<div align="center">

  <a href="https://channellabs.ai/">
    <picture><img alt="JudgeMaker logo" src="assets/logo.png" width="160px"></picture>
  </a>

  <p align="center">
    <h1>JudgeMaker</h1>
  </p>

  <p>
Optimize LLM-as-a-judge prompts to align with human preferences via reinforcement learning on synthetic data
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

### The Challenge with Current LLM Evaluation

LLM-as-a-judge has become the standard for evaluating language models at scale, but current approaches have significant limitations:

**Prompt Engineering Approach:**
- Relies on manual effort to craft effective prompts
- Difficult to systematically optimize for specific use cases
- Limited scalability and consistency

**Fine-Tuning Approach:**
- Requires extensive human preference data
- Creates black-box models with no transparency into judging criteria
- Expensive and time-consuming to implement

### Our Solution

JudgeMaker combines the advantages of both approaches. Instead of fine-tuning a model to judge responses directly, we fine-tune a model to generate judge prompts that align with human preferences. This maintains full transparency while achieving superior performance. In our experiments, human preference alignment increased from 64% with optimal prompt engineering to 68% with JudgeMaker.

#### Key Innovations

**🎭 Synthetic Conversations Backed by Detailed Personas**
We generate high-quality synthetic conversations by first creating diverse, detailed personas that represent your users. Each persona includes specific characteristics, scenarios, and use cases that enable an LLM to predict preferences accurately. This approach builds on established libraries like [DiaSynth](https://github.com/ntuspeechlab/DiaSynth) and our own [synthetic conversation generation](https://github.com/Channel-Labs/synthetic-conversation-generation) library.

**🔄 Reinforcement Learning with Verifiable Rewards**
Our breakthrough enables true reinforcement learning for preference alignment. By providing the reward model with full access to persona information while keeping the prompt-generating model blind to these details, we create verifiable rewards at scale. The model learns to generate prompts that help LLMs align with synthetic human preferences without ever seeing the underlying persona data.

### What This Repository Does

**Create Fine-Tuned Models**
Launch reinforcement learning fine-tuning jobs that train models to generate judge prompts matching your users' preferences.

**Evaluate Model Performance**
Test models against preference datasets to measure how effectively generated prompts enable LLMs to predict human preferences. Includes evaluation on [Chatbot Arena](https://lmarena.ai/leaderboard) conversations as an example.

---

*For detailed technical information about our approach, methodology, and experimental results, please refer to our technical white paper (COMING SOON!).*

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up JudgeMaker in your environment.

### Prerequisites

* Python 3.9 or higher
* pip (Python package manager)
* API keys for your chosen LLM provider:
  - [OpenAI API Key](https://platform.openai.com/docs/overview)

### Installation

1. Clone the repository
   ```bash
   git clone git@github.com:Channel-Labs/JudgeMaker.git
   ```
2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   ```

<!-- USAGE EXAMPLES -->
## Usage

This repository provides two main functionalities:

### 1. Fine-Tune a Judge Prompt Generation Model

Create a fine-tuned model using reinforcement learning to generate optimized LLM-as-a-judge prompts via the OpenAI API.

**Usage:**

```bash
python judge_maker/create_fine_tuning_job.py \
  --assistant-definition-file <ASSISTANT_DEFINITION_FILE> \
  --user-personas-file <USER_PERSONAS_FILE> \
  --conversations-file <CONVERSATIONS_FILE_NAME> \
  --num-attempts-per-conversation <NUM_ATTEMPTS> \
  --ground-truth-judge-model-id <GT_MODEL_ID> \
  --judge-prompt-generator-model-id <JPG_MODEL_ID> \
  --validation-fraction <VALIDATION_FRACTION>
```

**Arguments:**

- `--assistant-definition-file`: Path to the file containing your assistant definition (name and description)
- `--user-personas-file`: Path to the file containing your generated user personas
- `--conversations-file`: Path to the file containing conversations between the generated personas and your assistant
- `--num-attempts-per-conversation`: Number of ground truth scores to generate per conversation. Scores are averaged to calculate a single ground truth score per conversation (default: 5)
- `--ground-truth-judge-model-id`: Model ID for executing the ground truth grader judge (default: `o3`)
- `--judge-prompt-generator-model-id`: Model ID for generating candidate LLM-as-a-judge prompts during fine-tuning (default: `o4-mini`)
- `--validation-fraction`: Fraction of the dataset to withhold for validation during training (default: 0.25)

**Example:**

```bash
python judge_maker/create_fine_tuning_job.py \
  --assistant-definition-file "data/examples/assistant_definition.yaml" \
  --user-personas-file "data/examples/assistant_personas.yaml" \
  --conversations-file "data/examples/assistant_conversations.jsonl"
```

After execution, navigate to the [OpenAI fine-tuning dashboard](https://platform.openai.com/finetune) to monitor your job.

### 2. Evaluate Your Model

Measure how effectively your model generates judge prompts that enable LLMs to predict human preferences. This evaluation works with any preference dataset where humans choose between two AI responses. As a demonstration, we include the [Chatbot Arena dataset](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data). Our fine-tuned model improved accuracy for predicting multi-turn conversations from 64% to 68%. You can substitute your own preference datasets for domain-specific evaluation.

```bash
python judge_maker/evaluate_model.py \
  --assistant-definition-file <ASSISTANT_DEFINITION_FILE> \
  --evaluation-conversations-file <EVALUATION_CONVERSATIONS_FILE> \
  --output-file <OUTPUT_FILE> \
  --judge-prompt-generator-model-id <JPG_MODEL_ID> \
  --judge-model-id <JUDGE_MODEL_ID>
```

**Arguments:**

- `--assistant-definition-file`: Path to the file containing your assistant definition (name and description)
- `--evaluation-conversations-file`: Path to the file containing conversations to evaluate
- `--output-file`: Path where evaluation results will be saved
- `--judge-prompt-generator-model-id`: Model ID for generating judge prompts. Use your fine-tuned model ID to evaluate fine-tuning performance, or a base model ID to evaluate the baseline
- `--judge-model-id`: Model ID for judging conversations in the evaluation file (default: `o3`)

**Example:**

```bash
python judge_maker/evaluate_model.py \
  --assistant-definition-file "data/examples/assistant_definition.yaml" \
  --evaluation-conversations-file "data/evaluation/chatbot_arena_multi_turn_conversations.csv" \
  --output-file "evaluation_results.csv" \
  --judge-prompt-generator-model-id "o4-mini" 
```

<!-- CONTRIBUTING -->
## Contributing

Contributions make the open source community an amazing place to learn, inspire, and create. We **greatly appreciate** any contributions you make.

If you have suggestions for improvement, please fork the repository and create a pull request. Alternatively, feel free to start a discussion or open an issue here on GitHub—we'll review it promptly.

Don't forget to give the project a star! Thanks again!

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Created by [Channel Labs](https://channellabs.ai/)

Interested in understanding and improving your AI's behavior even further? Contact scott@channellabs.ai for any inquiries.
