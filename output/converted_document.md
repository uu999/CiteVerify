# BPMN ASSISTANT: AN LLM-BASED APPROACH TO BUSINESS 
PROCESS MODELING

## Josip Tomo Licardo

## Nikola Tanković

## Darko Etinger

Faculty of Informatics 
Juraj Dobrila University of Pula 
Zagrebačka 30 
52100 Pula, Croatia 
jlicardo@unipu.hr

Faculty of Informatics 
Juraj Dobrila University of Pula 
Zagrebacka 30 
52100 Pula, Croatia 
ntankov@unipu.hr

Faculty of Informatics 
Juraj Dobrila University of Pula 
Zagrebačka 30 
52100 Pula, Croaia 
detinger@unipu.hr

January 23, 2026

# ABSTRACT

This paper presents BPMN Assistant, a tool that leverages Large Language Models for natural 
language-based creation and editing of BPMN diagrams. While direct XML generation is common, it 
is verbose,slow,and prone to syntax errors during complex modifications. We introduce a specialized 
JSON-based intermediate representation designed to facilitate atomic editing operations through 
function calling. We evaluate our approach against direct XML manipulation using a suite of state-of- 
the-art models, including GPT-5.1, Claude 4.5 Sonnet, and DeepSeek V3. Results demonstrate that 
the JSON-based approach significantly outperforms direct XML in editing tasks,achieving higher 
or equivalent success rates across all evaluated models. Furthermore, despite requiring more input 
context,our approach reduces generation latency by approximately 43% and output token count by 
over 75%,offering a more reliable and responsive solution for interactive process modeling.

000210000

## 1 Introduction

Business Process Model and Notation (BPMN) [1] has long served as a standard for modeling business processes, 
enabling organizationsto visualize andoptimize theirworkflows.However,thecomplexityinherentincreating,editing, 
andinterpreting BPMN diagrams posessignificantchallenges,particularly forindividuals without specialized training 
[2]. As Recker [3] demonstrates through extensive research,BPMN's over-engineered nature and the prevalent lack 
of formal training among its users create substantial barriers to effective adoption. This bottleneck often reults in 
inefficiencies and a reliance on experts,which can increase operational costs and delay decision-making.

A significant challenge in modern organizations is the communication gap between IT departments and business 
stakeholders. While IT professionals are comfortable with formal modeling notations and technical specifications, 
business users typically express processes in natural language and informal descriptions. A systematic review by Njanka 
et al. [4] demonstrates that this disconnect leads to misunderstandings,requirements misalignment, and implementation 
delays,with communication barriers persisting despite significant investments in alignment initiatives.Furthermore, 
valuable process knowledge frequently remains trapped in the minds of domain experts or scattered across various 
informal documents, making it difficult to capture and formalize this expertise [5].

The complexity of extracting process models from unstructured text remains a significant challenge,as highlighted by 
Bellan et al.'s qualitative analysis [6]. Their research reveals that current methodologies primarily rely on ad-hoc rule- 
based approaches,which struggle with the complexity of real-world documents. This challenge is further compounded 
by the limitations of traditional process elicitation methods,as demonstrated by Baiao et al. [7], who propose a novel 
approachintegrating group storytelling techniques with textmining to capture the nuancesof human-centric activities 
in process modeling.

Recent research has highlighted the cognitive challenges faced by modelers when creating process models, particularly 
regarding cognitive load and task complexity. Weber et al. [8] introduced a psycho-physiological approach to assess

cognitive load during process modeling,utilizing real-timeeye movement analysis toidentify task-specific difficulties 
Their findings emphasize that process modeling requires substantial cognitive effort,especially in naming activities and 
managing complex structures. These insights reinforce the need for tools like BPMN Assistant,which aim to reduce 
cognitive barriers and enhance accessibility.

The technical barrier to BPMN usage, when combined with the dispersed nature of process documentation, creates a 
significant barrier to effective process management. Organizations often find themselves in a situation where valuable 
process knowledge exists but remains underutilized due to its informal and distributed nature.

Another critical issue is the dynamic nature of business processes in today's rapidly evolving business environment. 
Organizations need to frequenly update and modify their processes to remain competitive and adapt to changing market 
conditions [9]. However, the formal nature of BPMN and the expertise required to modify process models often create 
a bottleneck in implementing these changes, leading to a gap between actual business operations and their formal 
documentation. This challenge is evidenced by Leopold et al.'s analysis of 585 BPMN models from industry, which 
revealed persistent quality issues related to model complexity, particularly in areas such as splits and joins,message 
flows, and model decomposition, highlighting the technical expertise required for effective BPMN modeling [10].

The disconnect between formal process models and actual business operations is further exacerbated by the increasing 
complexity of modern business processes. These processes often span multiple departments, involve numerous 
stakeholders,and integrate with various systems and external partners. As van der Aalst and Weijters [11] demonstrate 
in their seminal work on process mining,organizations face significant challenges in accurately capturing and analyzing 
these complex processes,including issues with hidden tasks, duplicate activities, and non-free-choice constructs. The 
traditional modeling approaches struggle to address these challenges while maintaining accessibility for all stakeholders.

Prior to the emergence of Large Language Models (LLMs), numerous attempts were made to automate aspects of 
process modeling and bridge the natural language gap. Early work by Friedrich et al.[12] demonstrated the potential 
of rule-based approaches for extracting process models from natural language text, though these systems struggled 
with complex sentence structures and domain-specific terminology. The integration of natural language processing into 
process management faced significant challenges with semantic ambiguity and contextual understanding.Efforts by 
Leopold et al. [13] to automatically generate natural language descriptions from process models highlighted both the 
potential and limitations of traditional NLP approaches in process automation. These early automation attempts,while 
groundbreaking,were often constrained by their reliance on predefined rules and limited ability to handle variations in 
natural language expressions [14].

Advancements in artificial intelligence, particularly the emergence of LLMs,have opened new avenues for automating 
the creation and management of business process models [15, 16, 17]. These models have demonstrated exceptional 
capabilities in understanding and generating natural language, making them well-suited for bridging the gap between 
textual process descriptions and formal BPMN representations. Recent research by Klievtsova et al. [18] has shown 
that AI-assisted approaches to process modeling can be particularly effective,with AI-generated models often being 
preferred over those created by inexperienced human modelers. However, Rebmann et al. [19] highlight that while 
LLMs show promise in process-related tasks, their effectiveness often depends on proper fine-tuning and task-specific 
training, particularly for complex semantics-aware operations. This suggests significant potential for LLM-based 
tools in democratizing access to process modeling capabilities, while also emphasizing the importance of careful 
implementation and training approaches.

This work presents BPMN Assistant, a system that leverages LLMs to address these challenges. While recent 
research has demonstrated the potential of AI-assisted process modeling, the critical aspect of model modification and 
maintenance remains largely unexplored. Our work advances the state of the art by demonstrating the effectiveness of 
LLMs in editing existing BPMN diagrams through natural language instructions. Rather than framing BPMN modeling 
as a one-shot generation problem, this work is conceptualized around the idea of incremental process transformation. 
We argue that BPMN models should be treated as mutable process structures that can be reliably modified through a 
constrained set of well-defined operations,instead of being repeatedly regenerated in their entirety. This conceptual 
perspective motivates the separation of process logic from BPMN 2.0 XML syntax and forms the basis for the structured 
intermediate representation proposed in this study.

### 1.1 Research Objectives

This study addresses the following research questions:

·RQ1: How does a structured intermediate representation (JSON) compare to direct standard notation (XML) 
in terms of generation reliability and editing success rates?

•RQ2: To what extent does a function-based editing approach enable open-weights models to perform complex 
modeling tasks previously reserved for proprietary frontier models? 
• RQ3: How does the trade-off between increased input context and reduced output complexity impact overall 
system latency and operational efficiency?

### 1.2 Contribution

We propose a novel approach that utilizes a structured, JSON-based intermediate representation to abstract away the 
syntactic complexity of BPMN 2.0 XML. By shifting the generative task from producing verbose XML to manipulating 
a concise JSON structure, our system allows LLMs to focus on process logic rather than formatting rules. At a 
conceptual level, this approach reframes BPMN editing as a controlled transformation problem, where modifications 
are expressed as atomic logical operations rather than as low-level syntactic changes to XML documents.

Our evaluation reveals significant improvements in editing success rates compared to baseline approaches, particularly 
when using structured representations for model manipulation. Furthermore, we demonstrate that this approach 
significantly reduces processing latency and output token costs,making it a viable solution for interactive tools.

The implementation of BPMN Assistant is publicly available at https://github.com/jtlicardo/ 
bpmn-assistant.

The following sections detail our methodology, system implementation, and evaluation results, providing a foundation 
for future research in AI-assisted process modeling.

## 2 Related Work

The integration of LLMs into Business Process Management (BPM) has led to significant developments in process 
automation and modeling. The introduction of specialized tools like the BPMN-Chatbot [20] and ProMoAI [21] has 
demonstrated the practical applications of this technology. Initial explorations by Berti and Qafari[22] into using GPT-4 
and Bard (Gemini) for process mining tasks showed promising results in interpreting both procedural and declarative 
models, highlighting the potential of LLMs in the BPM domain. These developments have led to comprehensive 
frameworks for process modeling with LLMs [23],showing promising results in both automation and model quality.

### 2.1 Process Extraction and Prompting Strategies

In the area of process extraction, various approaches have demonstrated different methodologies and results. Ferreira 
et al. [24] developed a semi-automatic method for identifying business process elements from natural language texts, 
achieving 91.92% accuracy in their prototype implementation through carefully defined mapping rules. Takinga 
different approach, Neuberger et al. [25] introduced a universal prompting strategy that leverages LLMs for process 
model information extraction, demonstrating performance improvements of up to 8% F1 score over traditional machine 
learning methods.

The integration of prompt engineering in BPM has emerged as a promising direction, as discussed by Busch et al. 
[26]. Their research highlights how prompt engineering can effectively utilize pre-trained language models without 
extensive fine-tuning,addressing common challenges in process extraction and predictive monitoring. This approach is 
particularlyvaluableinscenarios withlimiteddataavailabilityoffering acomputationally sustainablesolutionforBPM 
applications.

### 2.2Interactive Modeling Tools

ProMoAI, introduced by Kourani et al. [21],represents a significant development in LLM-driven process modeling. 
The tool prompts an LLM to generate constrained Python code for constructing an intermediate POWL representation 
and incorporates prompt-engineering and iterative error-handling mechanisms to improve reliability. The resulting 
models are then transformed into standard notations and can be viewed and exported as BPMN and Petri nets (PNML).

The framework underpinning ProMoAI was evaluated in a subsequent benchmark study [27],which compared multiple 
LLMs using conformance checking (harmonic mean of fitness and precision against ground-truth event logs) as a 
model-quality measure. In this evaluation, Claude 3.5 Sonnet achieved the highest average quality score (0.93), 
approaching the ground-truth baseline (0.98).

Köpke and Safan [20] introduced the BPMN-Chatbot, a publicly available web-based tool designed for efficient 
LLM-based process modeling. The BPMN-Chatbot allows users to generate BPMN process models interactively using

text or voice input. It achieved notable efficiency gains,reducing token usage by up to 94% compared to alternative 
tools like ProMoAI while maintaining a correctness rate of 95%,which surpassed the best competitor's 86%. The 
authors emphasized the importance of user testing to evaluate the feedback loop's capabilities, which are critical for 
interactive process design.

While BPMN-Chatbot demonstrates notable strengths,particularly in token efficiency and correctness ratesits empirical 
evaluation does not isolate the robustness of iterative editing in the feedback loop (e.g,edit success rate over multi-step 
refinements,structural validity guarantees,or minimal-change behavior), beyond a preliminary technology acceptance 
test.

Hörner et al. [28] introduce BPMNGen, an LLM-based conversational framework for generating BPMN 2.0 pro- 
cess models from natural-language descriptions and iteratively refining them through user prompts. Their work is 
distinguished by a comprehensive human-centered evaluation, including controlled user studies assessing semantic 
alignment, cognitive load, acceptability, and comprehension performance of LLM-generated models compared to 
expert-created ones. The results indicate that LLM-generated models can achieve expert-level semantic quality for 
simple and moderately complex processes, while expert intervention remains beneficial for highly complex scenarios.

In contrast, BPMN Assistant focuses on a different aspect of the problem: robust and fine-grained interactive BPMN 
editing, emphasizing structural correctness,editing reliability, latency, and token efficiency across multiple LLM 
backends. Rather than evaluating model comprehensibility, our work addresses the engineering challenges of incremental 
model manipulation and validation, which are complementary to the human-centered quality dimensions studied in 
BPMNGen.

### 2.3Automated Generation and Evaluation Frameworks

In a comprehensive evaluation of process extraction approaches,Bellan et al. [29] analyzed ten state-of-the-art methods 
for extracting process models from textual descriptions. Their systematic comparison revealed significant variations 
in performance and methodology among existing tools, with no single approach achieving superior results across 
all evaluation metrics. This study highlighted the need for standardized evaluation frameworks and emphasized the 
challenges in automated process extraction that our work aims to address.

Kourani et al. [23] proposed a comprehensive framework leveraging LLMs to automate the generation and refinement of 
process models from textual descriptions. This framework demonstrated its ability to streamline process modeling tasks 
while maintaining sound and executable model outputs. The superiority of this approach was evident in its comparison 
with traditional methods, particularly in resolving errors and integrating user feedback effectively. GPT-4 showcased 
strong performance in generating process models, addressing errors, and adapting to user feedback, whereas Gemini 
struggled with similar tasks.

Nivon et al. [30] introduced a novel approach to automating BPMN process generation from textual requirements, 
addressing the challenges faced by non-expert users in process modeling. Their methodology employs a three-step 
approach utilizing a fine-tunedGPT-3.5model:first extracting tasksandordering constraintsfrom textual descriptions, 
then constructing an abstract syntax tree (AST) to reprsent task relationships,and finally converting the AST into 
a BPMN process. Their Java-based implementation achieved a 78.5% accuracy rate for valid BPMN processes in 
tests across 200 descriptions,demonstrating the viability of automated BPMN generation while highlighting areas for 
potential improvement through more advanced language models.

These studies collectively illustrate the transformative potential of LLMs in business process modeling. Tools like 
BPMN-Chatbot, ProMoAI,and Nivon et al.’s automated BPMN generator demonstrate the feasibility of combining 
generative AI with domain-specific methodologies to achieve higher efficiency,accuracy, and accessibility.Moreover 
the benchmarks and frameworks established by researchers provide a foundation for future advancements,emphasizing 
the importance of careful prompt design, user feedback integration, and domain-specific optimization in realizing the 
full potential of LLMs in BPM.

## 3 System Architecture

The system architecture is designed to operationalize the core conceptual assumption of this work: that reliable BPMN 
modeling with LLMs requires explicit separation between process semantics,editing logic, and concrete BPMN 
serialization. The BPMN Assistant system architecture, as illustrated in Figure 1,is designed to facilitate seamless 
interaction between users and BPMN models through natural language inputs. The system is composed of three primary 
components: the Python-based backend, the BPMN layout server, and the Vue.js frontend. Each component plays a 
critical role in ensuring the system's functionality,efficiency, and usability

[图片: User 
Frontend 
Backend 
LLM Service Layout Server 
(a) System architecture 
User Frontend Backend Service LLM Layout Server 
1. Request 
2. Process 
3. Query 
4. Response 
5.BPMNXML 
6. Enriched 
7. Visualization 
(b) Request sequence flow]

Figure 1: BPMN Assistant system: (a) Component architecture showing key system elements and (b) request sequence 
flow showing the temporal order of interactions.

| Axis | ProMoAI [21, 27] | BPMN-Chatbot [20] | BPMNGen [28] | BPMN Assistant |
| --- | --- | --- | --- | --- |
| Primary goal | NL→model generation + refinement/optimization | efficient NL→BPMN | NL→BPMN+ human-centered quality | reliable NL→BPMN + edit robustness |
| Output notation | BPMN + PNML | BPMN | BPMN | BPMN |
| Evaluation focus | conformance-based model quality | correctness + token efficiency + acceptance | semantic alignment + cognitive load + acceptability + comprehension | GED/RGED structural fidelity + failure rate |
| Evaluation method | automated | automated + user study | user study | automated |

Table 1: Comparison of interactive modeling tools by research objective and evaluation emphasis

### 3.1 Backend

The backend,implemented in Python, serves as the core computational engine of the system. Its primary responsibilities 
include handling user inputs,interacting with the LLM,managing BPMN diagrams,and providing APIs for the frontend. 
The backend leverages the FastAPI framework, chosen for its performance and simplicity in developing RESTful APIs.

The backend serves as the first point of contact for user inputs,analyzing these inputs to determine the user's intent. This 
can include conversational queries, such as asking questions about BPMN concepts, or operational commands, such as 
creating or modifying BPMN diagrams. When a conversational intent is recognized,the backend communicates with 
the LLM to generate a natural language response. If the input requires an operational response,the backend constructs 
a prompt to instruct the LLM to generate or modify a BPMN diagram. These diagrams are initially represented in 
JSON format and subsequently converted to BPMN XML for compatibility with visualization tools. Additionally, the 
backend supports the uploading of BPMN files that conform to the supported BPMN subset,enabling users to query or 
modify existing diagrams within the constraints of the system's JSON representation. The backend includes robust error 
handling and validation mechanisms to ensure that outputs from the LLM are accurate and reliable. If invalid JSON is 
generated, the backend retries the process or notifies the user of the error. By integrating with the BPMN layout server, 
the backend also ensures that diagrams are enriched with graphical information for enhanced visualization.

Model Integration and Selection Our system supports the usage of a wide range of LLMs from different providers, 
each chosen for its specific strengths and use cases. Table 2 provides an overview of the supported models and their 
capabilities.

| Provider | Model | Description |
| --- | --- | --- |
| OpenAI | GPT-5.1 | OpenAI’s flagship model for coding and agentic tasks. |
| OpenAI | GPT-5 mini GPT-40 | Faster, more cost-efficient version of GPT-5, used for well-defined tasks. Earlier-generation multimodal model with broad cross-modal capabilities [31]. |
| Anthropic | -- | -- |
| Anthropic | Claude 3.5 Sonnet Claude 4.5 Sonnet | Earlier-generation model with strong reasoning and long-context capabilities. Frontier model optimized for agentic reasoning and complex coding tasks. |
| Google | Gemini 2.0 Flash | Multimodal model with with low latency and a 1Mcontext window.[32] |
| Google | -- | -- |
| Fireworks AI | Llama 3.3 70B Qwen 2.5 72B | Meta's open-source text-only LLM [33]. Alibaba’s open-source model with strong multilingual capabilities [34]. |
| Fireworks AI | DeepSeek V3 | Open-source model specialized in technical and scientific reasoning [35]. |

Table 2: Overview of Integrated AI Models

Supported BPMN Elements The backend facilitates operations on a wide range of BPMN elements to support 
diverse modeling scenarios. In the category of tasks,the system supports generic tasks,user tasks,service tasks, send 
tasks, receive tasks, business rule tasks, manual tasks,and script tasks. To manage process flow logic, the system 
supports exclusive, parallel, and inclusive gateways.

The system also provides support for events,distinguishing between start, end, and intermediate types. Specifically, it 
supports generic,timer, and message start events; generic and message end events; as well as generic and message 
intermediate throw events, and generic,timer, and message intermediate catch events. This selection of elements 
enables the modeling of specific executable processes and event-driven workflows.

### 3.2 BPMN Layout Server

The BPMN layout server, implemented using Node.js with the Express.js framework,is dedicated to augmenting 
BPMN diagrams with graphical information. This server employs the bpmn-auto-layoutnpm library to generate DI 
(Diagram Interchange) information, which includes the graphical coordinates of BPMN elements.

The layout server takes BPMN XML files from the backend and enhances them by adding graphical coordinates 
which are essential for visual representation. Although the server efficiently enriches diagrams formost scenarios,it is 
currently limited in its ability to process multi-pool or multi-lane diagrams due to constraints in the bpmn-auto-layout 
library. The server is designed to work seamlessly with the backend, exchanging data through REST APIs to ensure 
smooth integration.

### 3.3 Frontend

The Vue.js-based frontend provides an intuitive graphical user interface (GUI) for user interaction. Its design focuses 
on accessibility, enabling non-experts to engage with BPMN modeling tasks effectively.

[图片: BPMN Assistant C 2 GPT-5.1 
ends the process. to BPMN rules If gou need furthermodifications orwant to Your BPuN process has beon updated as critical the fiow splis in paralone branch pageth Al garts of vour ne isuelizetheprocessletmeknow! rooesenowstartwhenanalertireivedres etis and thenchecs if the enoris critalis prooessspitthesystempagestheedinimmedietely AnindntrespesrtswnanalertisrivFs and the process endhs. aserigt nan ostemdionosis if theenaris tical the whilesimultaneouslywtingfor5minutesbefore trptinganautoatedsrvicestrtif therrisn criticalthesgm jut log the waming to the ctaae aeoonding dThe ÷□□□□□□□□□ + 区 × 
Mossage BPMN Asisa. 
Service statn 
n BPMN ASAP READT 
BPMNLyutSwr READY 
The apgioatinseeLuMe ane mey]

Figure 2: The web application interface featuring a dual-panel design: a chat interface on the left and a BPMN canvas 
on the right.

The frontend features a dual-panel design that facilitates interaction with both the LLM and BPMN diagrams. On the 
left,achatinterface allows users to submit natural language queries and view responses in real-time.Users can also 
select their preferred LLM for processing queries. On the right,a BPMN canvas, powered by bpmn.io (bpmn-js), 
displays the generated or modified BPMN diagrams. This canvas enables users to interact with the diagrams in a 
manner similar to traditional desktop tools. The frontend also provides feedback to users during diagram generation 
or modification, displaying status messages that indicate progress. Once a BPMN diagram is complete, users can 
download it for offline use or integration with other tools.

1https://github.com/bpmn-io/bpmn-auto-layout

The frontend bridges the gap between users and the underlying system, providing a user-friendly interface for BPMN 
modeling and interaction.

### 3.4 Data Flow

The system processes data through a structured sequence of interactions, as illustrated in Figure 1. The process initiates 
when a user submits a natural language request (Step 1). The frontend forwards this to the backend (Step 2), which 
performs intent recognition.If a modeling task is identified,the backend queries the LLM Service (Step 3).Unlike 
standard chatbots,the LLM is instructed to return structured JSON data representing the process or specific editing 
commands (Step 4). The backend then converts this JSON into standard BPMN 2.0 XML and transmits it to the Layout 
Server (Step 5). The layout server calculates the X/Y coordinates for the diagram elements and returns the enriched 
XML (Step 6).Finally, the backend delivers the renderable XML to the frontend for visualization (Step 7).

### 3.5 JSON Representation of BPMN Diagrams

The JSON intermediate representation is not merely an implementation convenience, but a conceptual abstraction that 
captures BPMN control-flow semantics independently of any concrete serialization format. BPMN Assistant utilizes a 
hierarchical JSON representation to abstract the verbose XML syntax of standard BPMN 2.0. This structure is designed 
to be easily generated by LLMs while maintaining sufficient fidelity to map back to valid XML. The representation uses 
a sequence of elements to describe the process flow. Unless branching logic is introduced via gateways,elements in the 
process array are executed sequentially. By elevating process structure to a first-class representation,this abstraction 
allows LLMs to reason about process logic directly, without being exposed to the syntactic complexity of BPMN XML.

Tasks Tasks represent atomic units of work. The system supports a comprehensive set of task types to handle 
various modeling scenarios. Beyond the generic task, the system supports: userTask, serviceTask, sendTask, 
receiveTask,businessRuleTask,manualTask,and scriptTask.

{

"type": "userTask",// or serviceTask,sendTask,etc. 
"id":"task_123", 
"label":"Approve request" 
}

Events The system distinguishes between start, end, and intermediate events. Crucially, it supports 
eventDefinition attributes to define specific triggers, such as timers or messages. This allows for the model- 
ing of event-driven architectures

{

"type":"intermediateCatchEvent", 
"id":"event_timer", 
"label":"Wait 24 hours"， 
"eventDefinition":"timerEventDefinition" 
}

Gateways Gateways manage flow divergence and convergence. The system supports exclusive (XOR), inclusive 
(OR), and parallel gateways.

A unique feature of our representation is the has_join boolean attribute. Since the JSON structure is hierarchical 
(nested), has_join:true explicitly signals the XML transformer that the branches merge backinto asingle flow after 
execution, triggering the generation of a converging gateway node. If false,the branches may end independently or 
loop back to previous elements.

To support cyclic flows (loops) within a nested structure, branches utilize an optional next field. This field contains the 
ID of a target element, allowing the flow to jump to any existing node in the process, breaking the strict hierarchy when 
necessary.

{

"type":"exclusiveGateway",// or "inclusiveGateway" 
"id":"gateway_1", 
"label":"Is approved?",

"has_join":true, 
"branches":[ 
{ 
"condition":"Yes", 
"path":[...]// Nested sequence of elements 
}， 
{ 
"condition":"No"， 
"path":□， 
"next":"task_start"// Example of aloop-back 
} 
]

}

For inclusive gateways,an is_default boolean field is available to designate the default flow path.

Parallel gateways are represented as an array of arrays, where each sub-array constitutes a concurrent execution path. 
Synchronization is handled implicitly: if the parent flow continues after the gateway,a converging parallel gateway is 
automatically generated in the XML output.

{ 
"type":"parallelGateway", 
"id":"parallel_1", 
"branches":[ 
[{"type"："task"，"label"："PathA"}]， 
[{"type":"task"，"label":"PathB"}] 
] 
}

This structured JSON representation serves as the backbone of the BPMN Assistant, allowing for the accurate and

efficient translation of user inputs into process models.

For a more complete example of the BPMN JSON representation, please refer to the Appendix (Section A).

### 3.6 Process Editing Functions

The system supports a set of specialized functions for modifying BPMN diagrams. These functions, outlined in 
Table 3,enable precise control over process elements while maintaining the structural integrity of the diagram. From 
a conceptual standpoint, these editing functions define the minimal set of operations required to express meaningful 
BPMN model transformations while preserving process soundness. When the LLM receives an editing request, it 
analyzes the natural language input and determines which function(s) to call to achieve the desired modifications.

| Function | Parameters | Description |
| --- | --- | --- |
| delete_element | element_id | Removes a specified element from the process |
| redirect_branch | branch_condition, next_id | Redirects a gateway branch to a new target element |
| add element | element, before _id*,after _id | Adds a new element at a specified position |
| move_element | element id.before id*.after id* | Relocates an existing element within the process |
| update_element | new element | Updates properties of an existing element |

*Optional parameters, only one should be provided

Table 3: Process Editing Functions

The LLM processes editing requests through a structured approach by first analyzing the user's natural language request 
to understand the desired changes. It then identifies the affected elements in the current process and determines the 
appropriate editing functions required to achieve the requested modification. Once identified, the model generates the 
function calls with the correct parameters and ensures that the proposed changes maintain process integrity through 
validation.

Each editing function is designed to perform a specific type of modification while preserving the process’s logical flow. 
For example,when deleting an element, the system automatically handles the reconnection of surrounding elements to 
maintain process continuity. Similarly, when adding new elements,the system ensures proper integration with existing 
process flows.

The granular nature of these functions allows the LLM to decompose complex editing requests into a series of atomic 
operations. This approach enhances reliability and makes it easier to validate and verify changes before they are applied 
to the process model.

### 3.7Validation and Soundness

To ensure the generation of syntactically sound BPMN models, the system incorporates a strict validation layer 
implemented in Python that intercepts the LLM output prior to XML conversion. This validator enforces core BPMN 
structural constraints,including the uniqueness of element identifiers,the correctness of connectivity between flow 
elements,the validity of gateway branch hierarchies,and the requirement that each process contains exactly one start 
event. When a violation is detected,the system initiates a self-correction loop in which the validation error is fed back to 
the LLM, prompting it to revise the intermediate representation. By acting as a programmatic guardrail, this validation 
mechanism ensures that the final BPMN XML is both syntactically valid and reliably interpretable by standard BPMN 
engines.

## 4 Evaluation

The evaluation of BPMN Assistant focuses on assessing its accuracy through Graph Edit Distance (GED) and Relative 
Graph Edit Distance (RGED), two widely recognized metrics for measuring process model similarity [36, 37, 38]. 
Traditional GED methods often struggle with gateway semantics and execution probabilities,limiting their effectiveness 
in BPMN similarity measurements [39]. Schoknecht et al. [40] provide a comprehensive review of process model 
similarity techniques, emphasizing that hybrid approaches—integrating syntactic, semantic,and behavioral compar- 
isons—tend to produce more reliable results. However, many of these methods require domain-specific adaptations to 
be effectivefor BPMN.

Our evaluation methodology deliberately focuses on structural and syntactic assessment through graph similarity 
measures. This approach was selected based on the research objectives of comparing intermediate representations for 
BPMN generation and modification. It is important to note that this evaluation does not encompass semantic evaluation 
approaches,such as those based on Petri nets or conformance checking from process mining. Similarly, we do not 
focus on execution performance through simulation studies, nor do we conduct comprehensive usability evaluations. 
While these aspects represent valuable dimensions for assessing process modeling tools,they fall outside the scope 
of the current research, which primarily examines the efficacy of different representation approaches in generating 
structurally accurate BPMN diagrams from natural language descriptions. The selected evaluation metrics allow for 
direct comparison between JSON and XML approaches while providing insight into the structural correctness of the 
generated models.

### 4.1 Dataset Composition

To ensure a robust evaluation across diverse scenarios, we constructed a dataset of 60 process descriptions spanning 20 
distinct business domains (e.g.,Marketing, Healthcare, Logistics). The dataset was curated to evaluate the system's 
ability to handle fundamental BPMN control-flow constructs,including sequential tasks, exclusive decision points 
(XOR-split/join), and concurrent execution paths (AND-split/join). This focus on core structural patterns provides a 
baseline for assessing the generation reliability of the underlying JSON representation.

The evaluation set was generated using a fixed prompt template and the OpenAI gpt-4.1 model. Each description 
was constrained to 7–8 activities and formulated in natural language while explicitly discouraging BPMN-specific 
terminology. Three descriptions were generated for each of the 20 business domains,after which the outputs were 
manually screened to remove ambiguous or degenerate cases.

Ground-truth BPMN diagrams were then created from these descriptions by multiple BPMN-trained annotators 
including the authors,following a consistent modeling guideline. All diagrams were subsequently reviewed and 
corrected by the authors to ensure syntactic validity and semantic alignment with the corresponding textual descriptions.

### 4.2 Graph Edit Distance (GED) and Relative Graph Edit Distance (RGED)

GED quantifies the cost of transforming one BPMN diagram into another by counting the minimum number of graph 
edit operations (node insertion,deletion,or substitution) required.To address the semantic variability inherent in natural 
language generation (e.g., an LLM generating "Process Order" versus "Handle Order"), we implemented a custom 
two-stage evaluation pipeline2.

First, we perform semantic label normalization. Raw BPMN XML files are parsed into a graph structure, and a 
lightweight LLM (GPT-5 mini) is employed to map semantically equivalent labels to identical abstract tokens (e.g., 
mapping both "Submit Order" and "Send Order" to token "A"). This ensures that the metric evaluates logical flow 
accuracy rather than penalizing surface-level lexical differences.

Second, we compute the GED using the NetworkX library with a specific cost configuration designed to distinguish 
between minor syntactic errors and major structural hallucinations. The cost functions are defined as follows:

The cost for inserting or deleting a node or edge is set to 1.0

Insertion and Deletion 
Node Substitution

The substitution cost csub(n1, n2) between two nodes is determined by:

(1)

## Edge Matching

Edges are matched based on the equality of their normalized labels.

The partial penalty of 0.5 for type mismatches allows us to distinguish cases where the model correctly infered the 
intent (the label) but selected the wrong BPMN element type (e.g.using a generic Task instead of a User Task)from 
cases where the model hallucinated an entirely incorrect step.

RGED normalizes the GED by considering the complexity of the involved graphs,providing a metric that is independent 
of diagram size:

(2)

(3)

where  represents an empty graph. This results in a similarity score where 1.0 represents a semantically identical 
structure and 0.0 represents total dissimilarity.

To implement the calculation, we utilized the NetworkX Python library, which provides efficient algorithms for comput- 
ing graph edit distance. Unlike prior approaches that rely on domain-specific heuristics or ML-based enhancements 
[38], our implementation applies standard GED operations to measure similarity between BPMN diagrams,albeit with 
the pre-processing step of semantic normalization.

By minimizing the edit distance to human-generated ground truth models, we utilize GED and RGED as proxies for 
structural fidelity and logical correctness. However,it is important to note that these metrics do not explicitly capture 
higher-level qualitative factors such as visual readability,layout simplicity,or cognitive load,which are emphasized in 
studies like those by Huang and Kumar [41] and Pavlicek et al. [42].

### 4.3 Comparison with Baseline XML Editing

To establish the effectiveness of BPMN Assistant,we conducted a comparative analysis between our atomic editing 
approach and a full-regeneration baseline (implemented here as direct XML generation). This baseline serves as a 
proxy forconversational refinementworkflows used by stateof-the-art tools such as BPMN-Chatbot[20] and ProMoAI 
[21]. Both systems prompt the LLM to produce a revised structured representation of the process conditioned on the 
current model state (e.g.,intermediate JSON in BPMN-Chatbot,or constrained code/POWL construction in ProMoAI). 
BPMN-Chatbotexplicitly targets tokenefficiency viaitsintermediate JSONrepresentation,whereas ProMoAI primarily 
leverages formal structure and constrained generation to improve reliability.

For the XML baseline, we utilized a system prompt instructing the model to act as a BPMN expert and output valid 
BPMN 2.0 XML directly,without intermediate steps or function calling. This represents the standard 'zero-shot'

2Implementation available at:https://github.com/jtlicardo/bpmn-ged

approach common in current LLM applications. For this comparison,we selected a representative set of diagram 
modification tasks and executed them using both representations. We evaluated the approaches using RGED and its 
derived similarity score, as defined in Equations (2) and (3).

### 4.4 Generation Accuracy

For the purposes of this evaluation,a "failure" is defined as any generation event that resulted in an unrenderable 
model. This encompasses both syntactic invalidity—such as unclosed XML tags or malformed JSON objects that 
prevent parsing—and structural inconsistencies,specifically reference hallucinations where sequence flows target node 
identifiers that do not exist within the element set. Any output falling into these categories was recorded as a failure in 
Tables 4 and 5 and treated as a non-functional result.

The comparative analysis revealed that our JSON-based representation achieved an average similarity score of 0.72 
compared to 0.70 for direct XML generation. While this indicates a slight numerical advantage for JSON,the difference 
is minimal, suggesting that both representations perform similarly in terms of structural accuracy. However, JSON 
demonstrated greater reliability, with fewer total failures across models.

| Model | JSON | XML | Failures (JSON) | Failures (XML) |
| --- | --- | --- | --- | --- |
| GPT-5.1 | 0.73 | 0.70 | 0 | 0 |
| GPT-5 mini | 0.77 | 0.68 | 0 | 0 |
| GPT-40 | 0.71 | 0.68 | 0 | 0 |
| Claude 4.5 Sonnet | 0.80 | 0.76 | 0 | 0 |
| Claude 3.5 Sonnet | 0.72 | 0.72 | 0 | 0 |
| Gemini 2.0 Flash | 0.70 | 0.66 | 0 | 0 |
| Llama 3.3 70B Instruct | 0.68 | 0.70 | 0 | 4 |
| Qwen 2.5 72B Instruct | 0.69 | 0.72 | 2 | 6 |
| DeepSeek V3 | 0.72 | 0.69 | 0 | 1 |

Table 4: Evaluation results showing similarity scores for different models using both our JSON-based BPMN represen- 
tation and direct XML generation approaches. Each model was evaluated on 60 BPMN generation tasks.

| Modality | Average Score | Total Failures |
| --- | --- | --- |
| JSON | 0.72 | 2 |
| XML | 0.70 | 11 |

Table 5: Average similarity scores and total failures per modality across all generated models.

Beyond similarity scores,JSON continues to outperform XML in terms of efficiency, a trend that holds even with newer, 
faster models. Latency measurements represent the total API response time recorded via the respective commercial 
providers (OpenAI, Anthropic,Google, and Fireworks AI). This metric reflects the real-world performance experienced 
by end-users of cloud-based LLM services. As shown in Table 6, JSON-based BPMN generation achieved a mean 
latency of 13.42 seconds compared to 24.82 seconds forXML.Additionally,while JSON requires a richer input prompt 
(averaging 2.678 tokens),it produces significantly more concise outputs (688 tokens vs 1,832 tokens). This reduction in 
output tokens is particularly advantageous for cost scaling,as output tokens are typically significantly more expensive 
than input tokens.Furthermore, the financial overhead of the increased input context is effectively mitigated by prompt 
caching technologies,which drastically reduce the cost of processing the static schema definitions and instructions 
required by our approach.

| Metric | JSON | XML |
| --- | --- | --- |
| Mean Latency (seconds) | 13.42 | 24.82 |
| Average Input Tokens | 2678.63 | 474.05 |
| Average Output Tokens | 688.04 | 1832.48 |

Table 6: Summary of latency and token usage between JSON and XML representations for BPMN generation.

[图片: JSON 
60 XML 
50 
  e 30 20 40 
10 
0 
Claude 4.5 Sonnet Claude 3.5 Sonnet Deep5eek V3 Gemini 2.0 Flash GPT-5.1 GPT-5 minI GPT-40 Lama 3 Qwen 2.5]

Figure 3: Latency comparison between JSON and XML-based BPMN generation.

### 4.5 Editing Capabilities

In addition to diagram generation, we evaluated the models' ability to interpret natural language editing instructions 
and perform the requested modifications. For this evaluation phase, we manually curated a dataset of 40 specific 
modification requests targeting various process elements. We employed a binary success/fail metric, as editing tasks 
involve correctly understanding the request and applying appropriate modifications to an existing diagram without 
corrupting the remaining structure.

To determine validity, we employed a two-stage verification process. First, a Python-based automated validator checked 
all outputs for syntactic correctness and referential integrity (e.g., ensuring no broken XML tags or references to non- 
existent IDs).Outputs failing this stage were automatically classified as failures.Second,the syntactically validoutputs 
underwent expert manual verification to assess semantic correctness. This step involved a binary check against the 
editing prompt (e.g.,verifying that a "delete task" operation actually removed the target node and correctly reconnected 
the surrounding sequence flows). Given the objective nature of these boolean operations,a single expert evaluator was 
deemed sufficient.

The results, summarized in Table 7,show that models consistently achieved higher success rates when using our JSON- 
based approach compared to direct XML manipulation. The performance gap is particularly evident in open-weight 
models; for instance, DeepSeek V3 achieved a 50% success rate with JSON but failed almost entirely with XML (8%). 
This suggests that the structured intermediate representation helps mitigate the complexities of verbose XML syntax, 
allowing models to focus on the logical changes rather than the syntactic overhead of the BPMN standard.

Even for flagship models like Claude 4.5 Sonnet,which achieved an 85% success rate in both modalities,the JSON 
approach offers superior programmatic verifiability.Because the JSON output adheres to a strict internal schema, the 
systemcanvalidate logicalconsistency(e.g.ensuringall branchtargetsexist)beforeconversion.Incontrast,validaing 
generated XML often requires complex parsing that may fail on minor syntax hallucinations.

As detailed in Table 8,the trade-off for this reliability is context size.The JSON editing approach requires supplying the 
full BPMN intermediate representation and various process examples in the prompt, resulting in an average of 22,071 
input tokens compared to 5,149 for XML. However, this 'up-front' cost pays dividends in speed: the JSON approach 
reduces generation latency by nearly 43% (20.35s vs 35.63s) and reduces output verbosity by over75% (607 vs 2.630 
tokens). In an interactive tool, this lower latency is critical for user experience.

## 5 Discussion

These findings empirically validate the initial conceptual assumption of this work,namely that constraining LLM 
interaction through structured representations leads to more reliable and controllable BPMN model manipulation.

[图片: JSON 
3000 XML 
2500 
eeeer 2000 1500 1000 
500 
0 
Claude 3.5 Sonnet DeepSeek V3 Gemini 2.0 Ftash GPT-5.1 GPT-5 Mini GPT-40 Qwen 2.5]

(a) Comparison of mean input tokens.

[图片: 2000 
JSON 
1750 XML 
ee  ee 1500 1250 1000 750 
500 
250 
0 
Claude 4.5 5onnet Cloude 3.5 Sonnet DeepSeek V3 Gemini 2.0 Flash GPT-5.1 GPT-5 Mini GPT-40 Liama 3 Qwen 2.5]

(b) Comparison of mean output tokens.

Figure 4: Token usage comparison for input and output tokens in JSON and XML-based BPMN generation. (a) shows 
the higher input token requirement for JSoN prompts. (b) shows the significantly more concise output generated by the 
JSON approach.

| Model | JSON | XML |
| --- | --- | --- |
| GPT-5.1 | 0.83 | 0.75 |
| GPT-5 mini | 0.73 | 0.53 |
| GPT-4o | 0.55 | 0.30 |
| Claude 4.5 Sonnet | 0.85 | 0.85 |
| Claude 3.5 Sonnet | 0.68 | 0.65 |
| Gemini 2.0 Flash | 0.45 | 0.33 |
| Llama 3.3 70B Instruct | 0.38 | 0.30 |
| Qwen 2.5 72B Instruct | 0.38 | 0.25 |
| DeepSeek V3 | 0.50 | 0.08 |

Table 7: Success rates for diagram editing based on natural language instructions,comparing our JSON-based approach 
with direct XML editing. Each model was evaluated on 40 diverse editing tasks.

[图片: ↑ +7.5% 
Llama 3.3 30.0% 37.5% 
1 +12.5% 
Qwen 2.5 25.0% 37.5% 
1 +12.5% 
Gemini 2.0 Flash 32.5% 45.0% 
T +42.5% 
Deepseek v3 7.5% 50.0% 
+25.0% 
GPT-40 30.0% 55.0% 
Claude 3.5 Sonnet 65.0% 67.5% 
+20.0% 
GPT-5 mini 52.5% 72.5% 
↑ +7.5% 
GPT-5.1 75.0% 82.5% 
Claude 4.5 Sonnet 85.0% 85.0% 
0 10 20 30 40 50 60 70 80 90 
Success Rate (%) 
JSON . XML]

Figure 5: Success rate comparison for diagram editing tasks between JSON and XML-based BPMN representations.

| Metric | JSON | XML |
| --- | --- | --- |
| Average Latency (s) | 20.35 | 35.63 |
| Average Input Tokens | 22,071.44 | 5,149.42 |
| Average Output Tokens | 607.92 | 2,630.44 |

Table 8: Comparative performance metrics between JSON and XML-based approaches for BPMN editing.

### 5.1 The Efficacy of Structured Representations (RQ1)

Our evaluation reveals a distinct dichotomy between generation and editing performance. In de novo generation tasks, 
the difference between JSON and XML approaches was marginal (e.g., GPT-5.1 achieved 0.73 with JSON vs 0.70 with 
XML). This suggests that for "tabula rasa" tasks,modern LLMs are sufficiently capable of managing XML syntax when 
generating from scratch.

However, the editing tasks expose the fragility of direct XML manipulation. The JSON-based approach achieved 
consistently higher success rates across all models. We attribute this difference to the verbosity and structural fragility 
of direct XML manipulation. When editing XML, an LLM must navigate verbose opening and closing tags, often 
losing track of the hierarchical structure or hallucinating invalid ID references. In contrast, our JSON schema and the 
associated atomic editing functions (e.g.,add_element)force the model to focus on the logical operation rather than 
the syntactic overhead. This structural constraint effectively acts as a guardrail,ensuring that modifications remain 
valid by design.

### 5.2Democratizing Process Modeling (RQ2)

Perhaps the most significant finding is the impact of our approach on open-weights models. While frontier proprietary 
models like Claude 4.5 Sonnet performed admirably in XML editing (85%),open-weights models such as DeepSeek 
V3 struggled significantly, achieving only an 8% success rate with XML. However,when switching to our JSON-based 
function calling approach, DeepSeek V3’s success rate jumped to 50%.

This has profound implications for enterprise adoption. Many organizations in regulated industries (finance, healthcare) 
cannot send sensitive process data to external APIs like OpenAI or Anthropic due to data privacy concerns. Our 
results demonstrate that by using a structured intermediate representation,organizations can deploy locally hosted 
open-weights models to perform complex process modeling tasks that were previously the domain ofmassive frontier 
models. This effectively lowers the barrier to entry for secure, on-premise AI process assistants.

### 5.3 Efficiency and Latency Trade-offs (RQ3)

The shift to a JSON-based editing workflow introduces a counter-intuitive trade-off: we drastically increase the input 
context size to decrease latency. As shown in Table 8,the JSON approach requires supplying the entire current stateof 
the process, leading to a ~ 4× increase in input tokens compared to the XML baseline.

However,in the current landscape of LLM economics and performance,this trade-off is highly favorable. Input 
tokens are computationally cheaper and faster to process than output tokens. By shifting the complexity to the input 
(the context) and restricting the output to concise JSoN function calls,we achieved a 43% reduction in total latency 
(20.35s vs 35.63s). For interactive tools where user experience depends on responsiveness,this reduction is critical. 
Furthermore,the 75% reduction in output tokens leads to significant cost savings at scale,as output tokens typically 
cost 3-4 times more than input tokens across major providers. This efficiency is further amplified by prompt caching, 
which allows the static JSON schema definitions—the bulk of our input context—to be processed at a fraction of the 
standard input cost.

### 5.4 Comparison with Existing Approaches

Unlike previous tools such as BPMN-Chatbot [20] and ProMoAI [21], which rely on regenerating the full process 
definition (via JSON or code) for every refinement, our system separates the editing logic from the model generation. 
While ProMoAI leverages formal methods (POWL) and BPMN-Chatbot utilizes intermediate JSON to optimize model 
creation, both approaches fundamentally treat modification as a conversational regeneration task. In contrast, BPMN 
Assistant implements a dedicated intermediate layer that supports atomic function calling, enabling targeted and 
deterministic updates without the overhead or instability associated with regenerating the entire process state for minor 
edits.

## 6 Limitations and Future Work

BPMN Assistant,while demonstrating promising results, has limitations that should be acknowledged. Although the 
system supports arobust setofelements,including intermediate events (timer,message) andinclusive gateways,the 
current implementation does not support collaboration diagrams (pools and lanes) or complex artifacts like data objects. 
This restricts the system’s application in scenarios requiring multi-participant modeling. However, as the focus of this 
work is on executable process orchestration, this omission does not impact the primary research objectives.

Due to the lack of semantic evaluation, the results should be considered experimental, as there is no guarantee that 
the business logic will be correctly integrated into the generated models. This limitation is particularly relevant when 
assessing the practical applicability of the generated diagrams in real-world scenarios.

The performance of BPMN Assistant is heavily dependent on the underlying LLM. Our evaluation revealed significant 
variations in performance across different models,with Claude 4.5 Sonnet and GPT-5.1 showing particularly strong 
results. Further investigation into the specific capabilities and limitations of different LLMs in the context of process 
modeling would provide valuable insights for future improvements.

Human-computer interaction (HCI) studies would be necessary to properly evaluate the usability of the system for 
non-technical users. While BPMN Assistant aims to lower the barrier to process modeling through natural language 
interaction, comprehensive user studies would be required to validate this claim.

Another area of concern is the reliance on clear and unambiguous natural language input. Users might encounter 
difficulties in achieving this clarity, especially in multilingual or context-specific scenarios where language and 
terminology nuances can introduce ambiguity.

It is important to note that this work deliberately does not focus on semantic evaluation (e.g.,using Petri nets)or 
conformance checking (e.g., process mining techniques), performance and execution time analysis (e.g.,simulation 
studies), or comprehensive usability evaluation. The approach used in this research was selected considering the 
research objectives,the method of model generation, and existing evaluation methods cited in the literature. Future work 
could address these limitations by expanding the range of supported BPMN elements,incorporating semantic evaluation

techniques, conducting usability studies,and optimizing the system for specific LLMs based on their strengths in 
process modeling tasks.

## 7 Conclusion

This paper presented BPMN Assistant, a novel approach to business process modeling that leverages LLMs to bridge 
the gap between natural language descriptions and formal BPMN representations. Through a comprehensive evaluation 
using state-of-the-art models, including GPT-5.1, Claude 4.5 Sonnet, and DeepSeek V3, we demonstrated that using a 
simplified JSONstructureinsteadofraw XMLyields significantbenefits inreliability andefficiency.

Our results indicate that while direct XML generation is feasible for creating simple models from scratch,it is often 
insufficient for iterative modification tasks. The proposed JSON-based approach outperformed or matched direct XML 
manipulation across all tested models in editing scenarios. While frontier models like Claude 4.5 Sonnet achieved parity 
in both modalities, the JSON advantage was critical for other models. This was most pronounced in open-weights 
models, where the structured approach enabled DeepSeek V3 to achieve a viable 50% success rate compared to a 
near-total failure (8%) with standard XML prompting. This suggests that intermediate representations are a key enabler 
for deploying cost-effective, locally hosted models in process automation workflows.

In terms of performance, the trade-off between increased input context and reduced output complexity proved highly 
advantageous. By shifting the computational burden to the input context,our approach reduced editing latency by 
approximately 43% and decreased output token usage by over 75%.These efficiency gains are critical for the practical 
adoption of LLM-based tools in interactive, real-time environments.

Looking ahead,future work will focus on conducting comprehensive usability studies to validate the system's effective- 
ness with non-technical business users. Additionally, we aim to further optimize the JSON schema to better handle 
complex cyclic dependencies and improve the system’s robustness when handling highly ambiguous natural language 
descriptions.

