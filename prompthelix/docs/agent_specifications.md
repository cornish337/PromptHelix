# Agent Specifications for PromptHelix

This document outlines the purpose, inputs, outputs, core functionalities, potential methodologies, high-level architecture, and interactions for each agent in the PromptHelix system.

---

## PromptArchitectAgent

### Purpose
Designs the initial genetic structure of prompts based on user requirements, system goals, or existing successful prompt patterns. It aims to create a diverse and potentially effective initial population of prompts for the genetic algorithm.

### Inputs
-   **Task Description**: The goal the prompt is meant to achieve (e.g., "generate a marketing email subject line," "translate English to Spanish," "write a Python function to sort a list").
-   **Keywords/Key Concepts**: Specific terms or ideas that should be included or focused on in the prompt.
-   **Constraints**: Any limitations or requirements for the prompt or its output (e.g., desired output length, complexity, format, target audience).
-   **Style Preferences**: Initial guidance on tone, voice, or style (potentially from user input or `StyleOptimizerAgent`).
-   **Learnings/Suggestions**: Feedback from `MetaLearnerAgent` on successful or unsuccessful prompt structures or components from previous runs.
-   **Domain-Specific Information**: Basic rules or common patterns from `DomainExpertAgent` for the given task type.

### Outputs
-   A `PromptChromosome` object (or a collection of them) representing an initial prompt. This includes the "genes" which are structured components of the prompt (e.g., instruction, context, examples, output format specifiers).
-   Metadata about the prompt design, such as the template used, reasoning for certain choices, etc.

### Core Functionalities
-   **Requirement Parsing**: Understand and interpret the input task description, keywords, and constraints.
-   **Template Selection/Generation**: Choose from a library of pre-defined prompt templates or dynamically generate a new structure.
-   **Gene Population**: Fill in the components (genes) of the selected prompt structure with initial content. This might involve incorporating keywords, formatting instructions, and placeholders.
-   **Initial Validation**: Perform basic checks for coherence, completeness, and adherence to constraints.
-   **Diversity Generation**: If creating multiple initial prompts, aim for structural and content diversity.

### Potential Algorithms/Methodologies
-   **Rule-Based Systems**: For template selection based on task type or keywords.
-   **Keyword-to-Fragment Mapping**: Associating input keywords with pre-defined prompt segments or phrases.
-   **Simple Natural Language Generation (NLG)**: Basic sentence construction for instructions or context.
-   **Combinatorial Generation**: Creating diverse prompts by combining different components in various ways.
-   **Case-Based Reasoning**: Adapting successful prompt structures from similar past tasks.
-   **Stub for ML Model**: Placeholder for future integration of a generative model to draft initial prompts.

### High-Level Architecture
1.  **Input Processor**: Normalizes and interprets input requirements.
2.  **Knowledge Base Accessor**: Retrieves templates, successful patterns, and suggestions from its internal knowledge or from other agents.
3.  **Structure Designer**: Selects or defines the overall architecture of the prompt.
4.  **Content Filler**: Populates the prompt structure with specific text and instructions.
5.  **Chromosome Formatter**: Assembles the components into a `PromptChromosome` object.

### Interaction with other agents
-   **Receives from**:
    -   `MetaLearnerAgent`: Suggestions for prompt design improvements.
    -   `StyleOptimizerAgent`: Initial style guidelines.
    -   `DomainExpertAgent`: Domain-specific constraints or common practices.
-   **Sends to**:
    -   **Genetic Algorithm Engine**: Provides the initial population of `PromptChromosome` objects.
    -   (Indirectly) `PromptCriticAgent` and `ResultsEvaluatorAgent` after prompts are used.

---

## PromptCriticAgent

### Purpose
Evaluates and critiques prompts based on their structure, content, and adherence to best practices, without necessarily executing them. It acts as a "static analyzer" for prompts.

### Inputs
-   A `PromptChromosome` object (or its string representation).
-   Potentially, the original task description or goals to provide context for the critique.
-   A set of heuristics, rules, or guidelines for good prompt design.
-   Feedback from `MetaLearnerAgent` on common pitfalls or effective patterns.

### Outputs
-   A critique score or a set of qualitative feedback points for the prompt (e.g., "instruction is ambiguous," "lacks sufficient context," "uses negative phrasing").
-   Suggestions for specific improvements to the prompt structure or wording.
-   Flags for potential issues (e.g., overly complex, too short, missing key components).

### Core Functionalities
-   **Structural Analysis**: Check for presence and completeness of key prompt components (e.g., clear instruction, context, examples).
-   **Clarity and Ambiguity Check**: Assess if the prompt's language is clear, concise, and unambiguous.
-   **Best Practice Adherence**: Verify against a checklist of known prompt engineering best practices (e.g., positive framing, clear output format).
-   **Constraint Violation Check**: Identify if the prompt violates any explicit constraints (though this might also be done by the system generating the prompt).
-   **Readability Assessment**: Evaluate how easy the prompt is to understand.

### Potential Algorithms/Methodologies
-   **Rule-Based Systems/Heuristics**: A primary method, using a predefined set of rules (e.g., "IF prompt length < X words THEN flag as too short").
-   **Simple NLP Techniques**:
    -   Keyword spotting for problematic phrases (e.g., "don't do X" vs. "do Y").
    -   Sentence complexity analysis.
    -   Readability scores (e.g., Flesch-Kincaid).
-   **Pattern Matching**: Identifying known anti-patterns or good patterns in prompt structures.
-   **Linter-like Functionality**: Similar to code linters, flagging stylistic or structural issues.
-   **Stub for ML Model**: Placeholder for a future ML model trained to predict prompt quality based on static features.

### High-Level Architecture
1.  **Prompt Parser**: Deconstructs the `PromptChromosome` into analyzable components.
2.  **Rule Engine**: Applies a set of predefined rules and heuristics.
3.  **NLP Analyzer**: Performs linguistic analysis on the prompt text.
4.  **Feedback Generator**: Compiles the findings into a structured critique and suggestions.

### Interaction with other agents
-   **Receives from**:
    -   **Genetic Algorithm Engine**: Receives prompts to be critiqued.
    -   `MetaLearnerAgent`: Updates to its rule set or heuristics based on system-wide learning.
-   **Sends to**:
    -   **Genetic Algorithm Engine**: Provides critique scores/feedback, which can be used as part of the fitness function or to guide mutation/crossover.
    -   `MetaLearnerAgent`: Reports on common issues found, contributing to system-wide learning.

---

## StyleOptimizerAgent

### Purpose
Refines prompts to enhance their style, tone, clarity, and persuasiveness, often based on specific target audience or desired communication effect. It focuses on the linguistic quality of the prompt itself.

### Inputs
-   A `PromptChromosome` object (or its string representation).
-   Target style parameters (e.g., "formal," "casual," "persuasive," "instructional," "empathetic").
-   Context about the task or target audience.
-   Feedback from `ResultsEvaluatorAgent` if previous stylistic choices led to poor outcomes.
-   Suggestions from `MetaLearnerAgent` on stylistic elements that correlate with good performance.

### Outputs
-   A modified `PromptChromosome` with improved style and phrasing.
-   A summary of changes made and the reasoning behind them.
-   Confidence score regarding the stylistic improvement.

### Core Functionalities
-   **Tone Analysis and Adjustment**: Modify wording to achieve a desired emotional or attitudinal tone.
-   **Clarity Enhancement**: Rephrase sentences for better understanding, reduce jargon if inappropriate.
-   **Persuasiveness Improvement**: Employ rhetorical techniques or persuasive language if applicable.
-   **Conciseness and Verbosity Management**: Adjust prompt length and detail level.
-   **Consistency Check**: Ensure consistent style and terminology throughout the prompt.

### Potential Algorithms/Methodologies
-   **Rule-Based Transformations**: Using synonym replacement, sentence restructuring rules based on style guides.
-   **Lexicon-Based Approaches**: Utilizing dictionaries of words associated with specific styles or tones.
-   **Simple NLP for Readability/Clarity**: Tools to simplify complex sentences or suggest clearer phrasings.
-   **Template-Based Refinement**: Applying stylistic templates to sections of the prompt.
-   **Stub for ML Models**:
    -   Paraphrasing models to generate stylistic variations.
    -   Style transfer models (more advanced).
-   **Sentiment Analysis Tools**: To gauge current tone and guide adjustments.

### High-Level Architecture
1.  **Prompt Analyzer**: Assesses the current style of the input prompt.
2.  **Style Target Interpreter**: Understands the desired stylistic output.
3.  **Transformation Engine**: Applies various linguistic rules and techniques to modify the prompt.
4.  **Evaluation Module**: (Optional) A simple internal check if the changes align with the target style.
5.  **Output Formatter**: Packages the optimized prompt.

### Interaction with other agents
-   **Receives from**:
    -   **Genetic Algorithm Engine / User**: Receives prompts needing stylistic optimization.
    -   `ResultsEvaluatorAgent`: Feedback on how style impacted previous prompt performance.
    -   `MetaLearnerAgent`: Insights into effective stylistic choices.
-   **Sends to**:
    -   **Genetic Algorithm Engine**: Provides the stylistically optimized prompt.
    -   `MetaLearnerAgent`: Reports on successful style transformations.

---

## ResultsEvaluatorAgent

### Purpose
Assesses the quality, relevance, and effectiveness of the outputs generated by an LLM in response to a given prompt. This evaluation is crucial for determining a prompt's fitness in the genetic algorithm.

### Inputs
-   The `PromptChromosome` that was used.
-   The output (response) generated by the LLM for that prompt.
-   The original task description and any defined success criteria or metrics (e.g., "output must be a valid JSON," "summary must be under 100 words and capture key points X, Y, Z").
-   Reference outputs or gold standards, if available.
-   Domain-specific evaluation criteria from `DomainExpertAgent`.

### Outputs
-   A fitness score for the prompt based on the quality of its output.
-   Detailed metrics (e.g., accuracy, relevance, coherence, adherence to constraints, factual correctness if checkable).
-   Error analysis or identification of specific failure modes in the LLM's output.
-   Categorization of the output (e.g., "successful," "partially successful," "failed").

### Core Functionalities
-   **Metric Calculation**: Compute quantitative scores based on defined metrics (e.g., ROUGE for summaries, BLEU for translations, code execution for code generation).
-   **Constraint Adherence Check**: Verify if the output meets all specified constraints (length, format, content).
-   **Content Analysis**: Assess relevance, coherence, and accuracy of the information in the output.
-   **Keyword/Concept Matching**: Check if required keywords or concepts are present in the output.
-   **Comparison with Reference**: If a gold standard is available, compare the LLM output against it.

### Potential Algorithms/Methodologies
-   **String Matching and Regex**: For checking specific keywords, formats, or patterns.
-   **Standard NLP Evaluation Metrics**: ROUGE, BLEU, METEOR, F1-score, precision, recall.
-   **Simple Semantic Similarity**: Using word embeddings (e.g., Word2Vec, GloVe, fastText) and cosine similarity to compare the output to the task description or reference answers.
-   **Rule-Based Checkers**: For validating output structure (e.g., JSON validation, code linters).
-   **Heuristic-Based Scoring**: Combining multiple checks into an overall fitness score.
-   **Stub for ML Classifiers**: Placeholder for ML models trained to classify output quality (e.g., a model predicting if a summary is good or bad).
-   **Human-in-the-loop Interface**: For tasks requiring subjective evaluation, provide an interface for human rating, which then feeds into the fitness score.

### High-Level Architecture
1.  **Input Processor**: Gathers the prompt, LLM output, and evaluation criteria.
2.  **Metric Calculation Engine**: Applies various automated evaluation techniques.
3.  **Constraint Verifier**: Checks against output constraints.
4.  **Scoring Module**: Aggregates results into a final fitness score and detailed feedback.
5.  **Feedback Formatter**: Structures the evaluation results.

### Interaction with other agents
-   **Receives from**:
    -   **LLM Interface / Genetic Algorithm Engine**: The prompt and its corresponding LLM output.
    -   `DomainExpertAgent`: Specific evaluation criteria or metrics relevant to the domain.
-   **Sends to**:
    -   **Genetic Algorithm Engine**: The fitness score and evaluation details, which directly influence the selection process.
    -   `MetaLearnerAgent`: Data about prompt performance, correlations between prompt features and output quality.
    -   Potentially to `StyleOptimizerAgent` or `PromptCriticAgent` if output failures suggest issues with prompt style or structure.

---

## MetaLearnerAgent

### Purpose
Analyzes the overall performance of the prompt generation and optimization process over time. It learns from successful and unsuccessful prompts and agents' actions to provide higher-level guidance, adapt strategies, and improve the efficiency of the entire PromptHelix system.

### Inputs
-   Historical data of prompts, their critiques (`PromptCriticAgent`), their style optimizations (`StyleOptimizerAgent`), their generated outputs, and their performance evaluations (`ResultsEvaluatorAgent`).
-   Logs of agent interactions and decisions.
-   Overall system goals or performance targets.
-   Feedback from other agents about persistent issues or successes.

### Outputs
-   **Strategic Recommendations**: Suggestions for modifying the genetic algorithm parameters (e.g., mutation rates, population size).
-   **Guidance for other Agents**:
    -   To `PromptArchitectAgent`: "Prompts with structure X tend to perform well for task Y."
    -   To `PromptCriticAgent`: "Consider adding a rule to flag prompts that frequently lead to Z error."
    -   To `StyleOptimizerAgent`: "Stylistic feature A seems to correlate with higher user engagement."
-   **Updated Heuristics/Rules**: Modifications to the rule sets used by other agents.
-   **Performance Reports**: Summaries of system performance, trends, and areas for improvement.
-   **Experiment Designs**: Suggestions for A/B testing different strategies or agent configurations.

### Core Functionalities
-   **Data Aggregation and Analysis**: Collect and process data from all parts of the system.
-   **Pattern Recognition**: Identify correlations between prompt characteristics, agent actions, and outcomes.
-   **Strategy Adaptation**: Adjust high-level parameters of the genetic algorithm or agent behaviors.
-   **Knowledge Base Management**: Maintain and update a central repository of learned insights.
-   **Performance Monitoring**: Track key performance indicators (KPIs) of the PromptHelix system.

### Potential Algorithms/Methodologies
-   **Statistical Analysis**: Correlation analysis, regression, hypothesis testing to find significant patterns.
-   **Simple Machine Learning**:
    -   Classification (e.g., predicting if a prompt will be successful based on its features).
    -   Clustering (e.g., grouping successful prompts to identify common archetypes).
    -   Reinforcement learning (more advanced, for optimizing agent policies).
-   **Rule Induction**: Automatically generating new rules for other agents based on observed data.
-   **Trend Analysis**: Identifying improvements or degradations in performance over time.
-   **Anomaly Detection**: Spotting unusual behavior or outcomes that warrant investigation.

### High-Level Architecture
1.  **Data Collector**: Gathers data from various agents and system logs.
2.  **Analytical Engine**: Performs statistical analysis and machine learning tasks.
3.  **Knowledge Synthesizer**: Translates findings into actionable insights and recommendations.
4.  **Strategy Adjuster**: Modifies system parameters or agent configurations.
5.  **Reporting Module**: Generates summaries and visualizations of system performance.

### Interaction with other agents
-   **Receives from (primarily)**:
    -   `PromptArchitectAgent`: Information about initial prompt designs.
    -   `PromptCriticAgent`: Critiques and identified prompt issues.
    -   `StyleOptimizerAgent`: Details of style changes and their intended effects.
    -   `ResultsEvaluatorAgent`: Fitness scores and detailed performance metrics of LLM outputs.
-   **Sends to (primarily)**:
    -   `PromptArchitectAgent`: Suggestions for better initial prompt structures or content.
    -   `PromptCriticAgent`: Recommendations for new critique rules or heuristics.
    -   `StyleOptimizerAgent`: Insights on effective stylistic elements.
    -   **Genetic Algorithm Engine**: Adjustments to GA parameters.
    -   System administrators/developers: Performance reports and insights.

---

## DomainExpertAgent

### Purpose
Provides domain-specific knowledge, constraints, terminology, and evaluation criteria to other agents, ensuring that prompts are tailored and effective for specific subject areas or tasks (e.g., medical, legal, coding in a particular language).

### Inputs
-   The current task or domain being addressed by the system.
-   Queries from other agents seeking specific domain information.
-   Access to domain-specific knowledge bases, ontologies, glossaries, or APIs.
-   (Potentially) User-provided domain expertise or configurations.

### Outputs
-   **Domain-Specific Constraints**: Rules that prompts or their outputs must adhere to (e.g., "In medical summaries, avoid speculative language").
-   **Key Terminology/Vocabulary**: Relevant jargon, keywords, or standard phrasing for the domain.
-   **Evaluation Criteria**: Specific metrics or quality dimensions that are important in the domain (e.g., "For legal documents, factual accuracy and citation correctness are paramount").
-   **Relevant Data Sources**: Pointers to data or information that should be used for context or examples in prompts.
-   **Validation Logic**: Snippets or rules to validate domain-specific aspects of LLM outputs.

### Core Functionalities
-   **Knowledge Retrieval**: Access and retrieve information from its configured knowledge sources.
-   **Constraint Definition**: Formulate and provide domain-specific rules.
-   **Terminology Provision**: Supply relevant vocabulary.
-   **Custom Evaluation Logic**: Offer methods or criteria for judging output quality in the specific domain.
-   **Query Answering**: Respond to requests for domain information from other agents.

### Potential Algorithms/Methodologies
-   **Knowledge Base Lookup**: Querying structured (e.g., SQL, SPARQL) or unstructured (e.g., document retrieval) knowledge sources.
-   **Rule-Based Systems**: For providing constraints and validation logic.
-   **Ontology Traversal**: If using formal ontologies to represent domain knowledge.
-   **Simple API Integration**: Connecting to external domain-specific APIs (e.g., a medical terminology service).
-   **Keyword Extraction/TF-IDF**: To identify key terms from domain corpora if not explicitly defined.
-   **Configuration Files**: Loading domain expertise from user-defined configuration files.

### High-Level Architecture
1.  **Query Interface**: Receives requests from other agents.
2.  **Knowledge Base Manager**: Interfaces with various internal or external knowledge sources.
3.  **Rule Engine**: Applies domain-specific rules to formulate responses or constraints.
4.  **Response Formatter**: Structures the output for the requesting agent.
5.  **Configuration Loader**: Loads domain profiles or settings.

### Interaction with other agents
-   **Receives from**:
    -   `PromptArchitectAgent`: Requests for domain-specific templates, keywords, or constraints during initial prompt design.
    -   `PromptCriticAgent`: Queries about domain-specific best practices or anti-patterns.
    -   `ResultsEvaluatorAgent`: Requests for domain-specific evaluation metrics or validation logic.
    -   `StyleOptimizerAgent`: Queries about domain-appropriate terminology or style.
-   **Sends to**:
    -   `PromptArchitectAgent`: Domain-specific guidelines, terminology.
    -   `PromptCriticAgent`: Domain-specific rules for prompt evaluation.
    -   `ResultsEvaluatorAgent`: Custom evaluation functions or critical quality aspects.
    -   `StyleOptimizerAgent`: Preferred domain vocabulary or phrasing.
    -   `MetaLearnerAgent`: Information about domain characteristics that might influence overall strategy.

---
