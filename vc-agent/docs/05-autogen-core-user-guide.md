# 05 — AutoGen Core User Guide (tam metin)

*Kaynak: [microsoft/autogen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html) · `python/docs/src/user-guide/core-user-guide/` · çekildi 2026-08-13 · 42 sayfanın tamamı*

Bu belge **birebir kopyadır, özet değildir.** Sayfalar rendered HTML'den değil,
reponun kendi `.md` / `.ipynb` kaynaklarından alındı; not defteri hücreleri
Markdown + kod bloğu olarak açıldı ve metin çıktıları korundu — bu sayfalarda
basılan çıktı anlatının parçası.

Telif MIT, microsoft/autogen. Bizim yorumumuz ve ölçümlerimiz için
[01](01-autogen-kaynak-haritasi.md), [02](02-autogen-el-kitabi.md) ve
[04 §3](04-vc-agentic-akis.md) — burada tek satır bile bize ait değil.

---

## İçindekiler

- **Getting started**
  - [Core](#core)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- **Core Concepts**
  - [Agent and Multi-Agent Applications](#agent-and-multi-agent-applications)
  - [Agent Runtime Environments](#agent-runtime-environments)
  - [Application Stack](#application-stack)
  - [Agent Identity and Lifecycle](#agent-identity-and-lifecycle)
  - [Topic and Subscription](#topic-and-subscription)
- **Framework Guide**
  - [Agent and Agent Runtime](#agent-and-agent-runtime)
  - [Message and Communication](#message-and-communication)
  - [Logging](#logging)
  - [Open Telemetry](#open-telemetry)
  - [Distributed Agent Runtime](#distributed-agent-runtime)
  - [Component config](#component-config)
- **Components Guide**
  - [Model Clients](#model-clients)
  - [Model Context](#model-context)
  - [Tools](#tools)
  - [Workbench (and MCP)](#workbench-and-mcp)
  - [Command Line Code Executors](#command-line-code-executors)
- **Multi-Agent Design Patterns**
  - [Intro](#intro)
  - [Concurrent Agents](#concurrent-agents)
  - [Sequential Workflow](#sequential-workflow)
  - [Group Chat](#group-chat)
  - [Handoffs](#handoffs)
  - [Mixture of Agents](#mixture-of-agents)
  - [Multi-Agent Debate](#multi-agent-debate)
  - [Reflection](#reflection)
  - [Code Execution](#code-execution)
- **Cookbook**
  - [Cookbook](#cookbook)
  - [Azure OpenAI with AAD Auth](#azure-openai-with-aad-auth)
  - [Termination using Intervention Handler](#termination-using-intervention-handler)
  - [User Approval for Tool Execution using Intervention Handler](#user-approval-for-tool-execution-using-intervention-handler)
  - [Extracting Results with an Agent](#extracting-results-with-an-agent)
  - [OpenAI Assistant Agent](#openai-assistant-agent)
  - [Using LangGraph-Backed Agent](#using-langgraph-backed-agent)
  - [Using LlamaIndex-Backed Agent](#using-llamaindex-backed-agent)
  - [Local LLMs with LiteLLM & Ollama](#local-llms-with-litellm-ollama)
  - [Instrumentating your code locally](#instrumentating-your-code-locally)
  - [Topic Subscription Scenarios](#topic-subscription-scenarios)
  - [Structured output using GPT-4o models](#structured-output-using-gpt-4o-models)
  - [Tracking LLM usage with a logger](#tracking-llm-usage-with-a-logger)
- **FAQs**
  - [FAQs](#faqs)

---

# Getting started


## Core

*[index.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html)*

---
myst:
  html_meta:
    "description lang=en": |
      User Guide for AutoGen Core, a framework for building multi-agent applications with AI agents.
---

### Core

```{toctree}
:maxdepth: 1
:hidden:

installation
quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Core Concepts

core-concepts/agent-and-multi-agent-application
core-concepts/architecture
core-concepts/application-stack
core-concepts/agent-identity-and-lifecycle
core-concepts/topic-and-subscription
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Framework Guide

framework/agent-and-agent-runtime
framework/message-and-communication
framework/logging
framework/telemetry
framework/distributed-agent-runtime
framework/component-config
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Components Guide

components/model-clients
components/model-context
components/tools
components/workbench
components/command-line-code-executors
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Multi-Agent Design Patterns

design-patterns/intro
design-patterns/concurrent-agents
design-patterns/sequential-workflow
design-patterns/group-chat
design-patterns/handoffs
design-patterns/mixture-of-agents
design-patterns/multi-agent-debate
design-patterns/reflection
design-patterns/code-execution-groupchat
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: More

cookbook/index
faqs
```

AutoGen core offers an easy way to quickly build event-driven, distributed, scalable, resilient AI agent systems. Agents are developed by using the [Actor model](https://en.wikipedia.org/wiki/Actor_model). You can build and run your agent system locally and easily move to a distributed system in the cloud when you are ready.

Key features of AutoGen core include:

```{gallery-grid}
:grid-columns: 1 2 2 3

- header: "{fas}`network-wired;pst-color-primary` Asynchronous Messaging"
  content: "Agents communicate through asynchronous messages, enabling event-driven and request/response communication models."
- header: "{fas}`cube;pst-color-primary` Scalable & Distributed"
  content: "Enable complex scenarios with networks of agents across organizational boundaries."
- header: "{fas}`code;pst-color-primary` Multi-Language Support"
  content: "Python & Dotnet interoperating agents today, with more languages coming soon."
- header: "{fas}`globe;pst-color-primary` Modular & Extensible"
  content: "Highly customizable with features like custom agents, memory as a service, tools registry, and model library."
- header: "{fas}`puzzle-piece;pst-color-primary` Observable & Debuggable"
  content: "Easily trace and debug your agent systems."
- header: "{fas}`project-diagram;pst-color-primary` Event-Driven Architecture"
  content: "Build event-driven, distributed, scalable, and resilient AI agent systems."
```



## Installation

*[installation.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/installation.html)*

### Installation

#### Create a Virtual Environment (optional)

When installing AgentChat locally, we recommend using a virtual environment for the installation. This will ensure that the dependencies for AgentChat are isolated from the rest of your system.

``````{tab-set}

`````{tab-item} venv

Create and activate:

Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows command-line:
```batch
python3 -m venv .venv
.venv\Scripts\activate.bat
```

To deactivate later, run:

```bash
deactivate
```

`````

`````{tab-item} conda

[Install Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) if you have not already.


Create and activate:

```bash
conda create -n autogen python=3.12
conda activate autogen
```

To deactivate later, run:

```bash
conda deactivate
```


`````



``````

#### Install using pip

Install the `autogen-core` package using pip:

```bash

pip install "autogen-core"
```

```{note}
Python 3.10 or later is required.
```

#### Install OpenAI for Model Client

To use the OpenAI and Azure OpenAI models, you need to install the following
extensions:

```bash
pip install "autogen-ext[openai]"
```

If you are using Azure OpenAI with AAD authentication, you need to install the following:

```bash
pip install "autogen-ext[azure]"
```

#### Install Docker for Code Execution (Optional)

We recommend using Docker to use {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` for execution of model-generated code.
To install Docker, follow the instructions for your operating system on the [Docker website](https://docs.docker.com/get-docker/).

To learn more code execution, see [Command Line Code Executors](./components/command-line-code-executors.ipynb)
and [Code Execution](./design-patterns/code-execution-groupchat.ipynb).



## Quick Start

*[quickstart.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/quickstart.html)*

### Quick Start

:::{note}
See [here](installation) for installation instructions.
:::

Before diving into the core APIs, let's start with a simple example of two agents that count down from 10 to 1.

We first define the agent classes and their respective procedures for 
handling messages.
We create two agent classes: `Modifier` and `Checker`. The `Modifier` agent modifies a number that is given and the `Check` agent checks the value against a condition.
We also create a `Message` data class, which defines the messages that are passed between the agents.

```python
from dataclasses import dataclass
from typing import Callable

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler


@dataclass
class Message:
    content: int


@default_subscription
class Modifier(RoutedAgent):
    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self._modify_val = modify_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        val = self._modify_val(message.content)
        print(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())  # type: ignore


@default_subscription
class Checker(RoutedAgent):
    def __init__(self, run_until: Callable[[int], bool]) -> None:
        super().__init__("A checker agent.")
        self._run_until = run_until

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if not self._run_until(message.content):
            print(f"{'-'*80}\nChecker:\n{message.content} passed the check, continue.")
            await self.publish_message(Message(content=message.content), DefaultTopicId())
        else:
            print(f"{'-'*80}\nChecker:\n{message.content} failed the check, stopping.")
```

You might have already noticed, the agents' logic, whether it is using model or code executor,
is completely decoupled from
how messages are delivered. This is the core idea: the framework provides
a communication infrastructure, and the agents are responsible for their own
logic. We call the communication infrastructure an **Agent Runtime**.

Agent runtime is a key concept of this framework. Besides delivering messages,
it also manages agents' lifecycle. 
So the creation of agents are handled by the runtime.

The following code shows how to register and run the agents using 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
a local embedded agent runtime implementation.

```{note}
If you are using VSCode or other Editor remember to import asyncio and wrap the code with async def main() -> None: and run the code with asyncio.run(main()) function.
```

```python
from autogen_core import AgentId, SingleThreadedAgentRuntime

# Create a local embedded runtime.
runtime = SingleThreadedAgentRuntime()

# Register the modifier and checker agents by providing
# their agent types, the factory functions for creating instance and subscriptions.
await Modifier.register(
    runtime,
    "modifier",
    # Modify the value by subtracting 1
    lambda: Modifier(modify_val=lambda x: x - 1),
)

await Checker.register(
    runtime,
    "checker",
    # Run until the value is less than or equal to 1
    lambda: Checker(run_until=lambda x: x <= 1),
)

# Start the runtime and send a direct message to the checker.
runtime.start()
await runtime.send_message(Message(10), AgentId("checker", "default"))
await runtime.stop_when_idle()
```

```text
--------------------------------------------------------------------------------
Checker:
10 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 10 to 9
--------------------------------------------------------------------------------
Checker:
9 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 9 to 8
--------------------------------------------------------------------------------
Checker:
8 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 8 to 7
--------------------------------------------------------------------------------
Checker:
7 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 7 to 6
--------------------------------------------------------------------------------
Checker:
6 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 6 to 5
--------------------------------------------------------------------------------
Checker:
5 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 5 to 4
--------------------------------------------------------------------------------
Checker:
4 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 4 to 3
--------------------------------------------------------------------------------
Checker:
3 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 3 to 2
--------------------------------------------------------------------------------
Checker:
2 passed the check, continue.
--------------------------------------------------------------------------------
Modifier:
Modified 2 to 1
--------------------------------------------------------------------------------
Checker:
1 failed the check, stopping.
```

From the agent's output, we can see the value was successfully decremented from 10 to 1 as the modifier and checker conditions dictate.

AutoGen also supports a distributed agent runtime, which can host agents running on
different processes or machines, with different identities, languages and dependencies.

To learn how to use agent runtime, communication, message handling, and subscription, please continue
reading the sections following this quick start.


---

# Core Concepts


## Agent and Multi-Agent Applications

*[core-concepts/agent-and-multi-agent-application.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-and-multi-agent-application.html)*

### Agent and Multi-Agent Applications

An **agent** is a software entity that communicates via messages, maintains its own state, and performs actions in response to received messages or changes in its state. These actions may modify the agent’s state and produce external effects, such as updating message logs, sending new messages, executing code, or making API calls.

Many software systems can be modeled as a collection of independent agents that interact with one another. Examples include:

- Sensors on a factory floor
- Distributed services powering web applications
- Business workflows involving multiple stakeholders
- AI agents, such as those powered by language models (e.g., GPT-4), which can write code, interface with external systems, and communicate with other agents.

These systems, composed of multiple interacting agents, are referred to as **multi-agent applications**.

> **Note:**  
> AI agents typically use language models as part of their software stack to interpret messages, perform reasoning, and execute actions.

#### Characteristics of Multi-Agent Applications

In multi-agent applications, agents may:

- Run within the same process or on the same machine
- Operate across different machines or organizational boundaries
- Be implemented in diverse programming languages and make use of different AI models or instructions
- Work together towards a shared goal, coordinating their actions through messaging

Each agent is a self-contained unit that can be developed, tested, and deployed independently. This modular design allows agents to be reused across different scenarios and composed into more complex systems.

Agents are inherently **composable**: simple agents can be combined to form complex, adaptable applications, where each agent contributes a specific function or service to the overall system.



## Agent Runtime Environments

*[core-concepts/architecture.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/architecture.html)*

### Agent Runtime Environments

At the foundation level, the framework provides a _runtime environment_, which facilitates
communication between agents, manages their identities and lifecycles,
and enforce security and privacy boundaries.

It supports two types of runtime environment: _standalone_ and _distributed_.
Both types provide a common set of APIs for building multi-agent applications,
so you can switch between them without changing your agent implementation.
Each type can also have multiple implementations.

#### Standalone Agent Runtime

Standalone runtime is suitable for single-process applications where all agents
are implemented in the same programming language and running in the same process.
In the Python API, an example of standalone runtime is the {py:class}`~autogen_core.SingleThreadedAgentRuntime`.

The following diagram shows the standalone runtime in the framework.

![Standalone Runtime](architecture-standalone.svg)

Here, agents communicate via messages through the runtime, and the runtime manages
the _lifecycle_ of agents.

Developers can build agents quickly by using the provided components including
_routed agent_, AI model _clients_, tools for AI models, code execution sandboxes,
model context stores, and more.
They can also implement their own agents from scratch, or use other libraries.

#### Distributed Agent Runtime

Distributed runtime is suitable for multi-process applications where agents
may be implemented in different programming languages and running on different
machines.

![Distributed Runtime](architecture-distributed.svg)

A distributed runtime, as shown in the diagram above,
consists of a _host servicer_ and multiple _workers_.
The host servicer facilitates communication between agents across workers
and maintains the states of connections.
The workers run agents and communicate with the host servicer via _gateways_.
They advertise to the host servicer the agents they run and manage the agents' lifecycles.

Agents work the same way as in the standalone runtime so that developers can
switch between the two runtime types with no change to their agent implementation.



## Application Stack

*[core-concepts/application-stack.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/application-stack.html)*

### Application Stack

AutoGen core is designed to be an unopinionated framework that can be used to build
a wide variety of multi-agent applications. It is not tied to any specific
agent abstraction or multi-agent pattern.

The following diagram shows the application stack.

![Application Stack](application-stack.svg)

At the bottom of the stack is the base messaging and routing facilities that
enable agents to communicate with each other. These are managed by the
agent runtime, and for most applications, developers only need to interact
with the high-level APIs provided by the runtime (see [Agent and Agent Runtime](../framework/agent-and-agent-runtime.ipynb)).

At the top of the stack, developers need to define the
types of the messages that agents exchange. This set of message types
forms a behavior contract that agents must adhere to, and the
implementation of the contracts determines how agents handle messages.
The behavior contract is also sometimes referred to as the message protocol.
It is the developer's responsibility to implement the behavior contract.
Multi-agent patterns emerge from these behavior contracts
(see [Multi-Agent Design Patterns](../design-patterns/intro.md)).

#### An Example Application

Consider a concrete example of a multi-agent application for
code generation. The application consists of three agents:
Coder Agent, Executor Agent, and Reviewer Agent.
The following diagram shows the data flow between the agents,
and the message types exchanged between them.

![Code Generation Example](code-gen-example.svg)

In this example, the behavior contract consists of the following:

- `CodingTaskMsg` message from application to the Coder Agent
- `CodeGenMsg` from Coder Agent to Executor Agent
- `ExecutionResultMsg` from Executor Agent to Reviewer Agent
- `ReviewMsg` from Reviewer Agent to Coder Agent
- `CodingResultMsg` from the Reviewer Agent to the application

The behavior contract is implemented by the agents' handling of these messages. For example, the Reviewer Agent listens for `ExecutionResultMsg`
and evaluates the code execution result to decide whether to approve or reject,
if approved, it sends a `CodingResultMsg` to the application,
otherwise, it sends a `ReviewMsg` to the Coder Agent for another round of
code generation.

This behavior contract is a case of a multi-agent pattern called _reflection_,
where a generation result is reviewed by another round of generation,
to improve the overall quality.



## Agent Identity and Lifecycle

*[core-concepts/agent-identity-and-lifecycle.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-identity-and-lifecycle.html)*

(agentid_and_lifecycle)=
### Agent Identity and Lifecycle

The agent runtime manages agents' identities
and lifecycles.
Application does not create agents directly, rather,
it registers an agent type with a factory function for
agent instances.
In this section, we explain how agents are identified
and created by the runtime.

#### Agent ID

Agent ID uniquely identifies an agent instance within
an agent runtime -- including distributed runtime.
It is the "address" of the agent instance for receiving messages.
It has two components: agent type and agent key.

```{note}
Agent ID = (Agent Type, Agent Key)
```

The agent type is not an agent class.
It associates an agent with a specific
factory function, which produces instances of agents
of the same agent type.
For example, different factory functions can produce the same
agent class but with different constructor parameters.
The agent key is an instance identifier
for the given agent type.
Agent IDs can be converted to and from strings. the format of this string is:
```{note}
Agent_Type/Agent_Key
```
Types and Keys are considered valid if they only contain alphanumeric letters (a-z) and (0-9), or underscores (_). A valid identifier cannot start with a number, or contain any spaces.

In a multi-agent application, agent types are
typically defined directly by the application, i.e., they
are defined in the application code.
On the other hand, agent keys are typically generated given
messages delivered to the agents, i.e., they are defined
by the application data.

For example, a runtime has registered the agent type `"code_reviewer"`
with a factory function producing agent instances that perform
code reviews. Each code review request has a unique ID `review_request_id`
to mark a dedicated
session.
In this case, each request can be handled by a new instance
with an agent ID, `("code_reviewer", review_request_id)`.

#### Agent Lifecycle

When a runtime delivers a message to an agent instance given its ID,
it either fetches the instance,
or creates it if it does not exist.

![Agent Lifecycle](agent-lifecycle.svg)

The runtime is also responsible for "paging in" or "out" agent instances
to conserve resources and balance load across multiple machines.
This is not implemented yet.



## Topic and Subscription

*[core-concepts/topic-and-subscription.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/topic-and-subscription.html)*

### Topic and Subscription

There are two ways for runtime to deliver messages,
direct messaging or broadcast. Direct messaging is one to one: the sender
must provide the recipient's agent ID. On the other hand,
broadcast is one to many and the sender does not provide recipients'
agent IDs.

Many scenarios are suitable for broadcast.
For example, in event-driven workflows, agents do not always know who
will handle their messages, and a workflow can be composed of agents
with no inter-dependencies.
This section focuses on the core concepts in broadcast: topic and subscription.

(topic_and_subscription_topic)=
#### Topic

A topic defines the scope of a broadcast message.
In essence, agent runtime implements a publish-subscribe model through
its broadcast API: when publishing a message, the topic must be specified.
It is an indirection over agent IDs.

A topic consists of two components: topic type and topic source.

```{note}
Topic = (Topic Type, Topic Source)
```

Similar to [agent ID](./agent-identity-and-lifecycle.md#agent-id),
which also has two components, topic type is usually defined by
application code to mark the type of messages the topic is for.
For example, a GitHub agent may use `"GitHub_Issues"` as the topic type
when publishing messages about new issues.

Topic source is the unique identifier for a topic within a topic type.
It is typically defined by application data.
For example, the GitHub agent may use `"github.com/{repo_name}/issues/{issue_number}"`
as the topic source to uniquely identifies the topic.
Topic source allows the publisher to limit the scope of messages and create
silos.

Topic IDs can be converted to and from strings. the format of this string is:
```{note}
Topic_Type/Topic_Source
```
Types are considered valid if they are in UTF8 and only contain alphanumeric letters (a-z) and (0-9), or underscores (_). A valid identifier cannot start with a number, or contain any spaces.
Sources are considered valid if they are in UTF8 and only contain characters between (inclusive) ascii 32 (space) and 126 (~).

#### Subscription

A subscription maps topic to agent IDs.

![Subscription](subscription.svg)

The diagram above shows the relationship between topic and subscription.
An agent runtime keeps track of the subscriptions and uses them to deliver
messages to agents.

If a topic has no subscription, messages published to this topic will
not be delivered to any agent.
If a topic has many subscriptions, messages will be delivered
following all the subscriptions to every recipient agent only once.
Applications can add or remove subscriptions using agent runtime's API.

#### Type-based Subscription

A type-based subscription maps a topic type to an agent type
(see [agent ID](./agent-identity-and-lifecycle.md#agent-id)).
It declares an unbounded mapping from topics to agent IDs without knowing the
exact topic sources and agent keys.
The mechanism is simple: any topic matching the type-based subscription's
topic type will be mapped to an agent ID with the subscription's agent type
and the agent key assigned to the value of the topic source.
For Python API, use {py:class}`~autogen_core.components.TypeSubscription`.

```{note}
Type-Based Subscription = Topic Type --> Agent Type
```

Generally speaking, type-based subscription is the preferred way to declare
subscriptions. It is portable and data-independent:
developers do not need to write application code that depends on specific agent IDs.

##### Scenarios of Type-Based Subscription

Type-based subscriptions can be applied to many scenarios when the exact
topic or agent IDs are data-dependent.
The scenarios can be broken down by two considerations:
(1) whether it is single-tenant or multi-tenant, and
(2) whether it is a single topic or multiple topics per tenant.
A tenant typically refers to a set of agents that handle a specific 
user session or a specific request. 

###### Single-Tenant, Single Topic

In this scenario, there is only one tenant and one topic for the entire
application.
It is the simplest scenario and can be used in many cases
like a command line tool or a single-user application.

To apply type-based subscription for this scenario, create one type-based
subscription for each agent type, and use the same topic type for all
the type-based subscriptions.
When you publish, always use the same topic, i.e., the same topic type and topic source.

For example, assuming there are three agent types: `"triage_agent"`,
`"coder_agent"` and `"reviewer_agent"`, and the topic type is `"default"`,
create the following type-based subscriptions:

```python
# Type-based Subscriptions for single-tenant, single topic scenario
TypeSubscription(topic_type="default", agent_type="triage_agent")
TypeSubscription(topic_type="default", agent_type="coder_agent")
TypeSubscription(topic_type="default", agent_type="reviewer_agent")
```

With the above type-based subscriptions, use the same topic source 
`"default"` for all messages. So the topic is always `("default", "default")`.
A message published to this topic will be delivered to all the agents of
all above types. Specifically, the message will be sent to the following agent IDs:

```python
# The agent IDs created based on the topic source
AgentID("triage_agent", "default")
AgentID("coder_agent", "default")
AgentID("reviewer_agent", "default")
```

The following figure shows how type-based subscription works in this example.

![Type-Based Subscription Single-Tenant, Single Topic Scenario Example](type-subscription-single-tenant-single-topic.svg)

If the agent with the ID does not exist, the runtime will create it.


###### Single-Tenant, Multiple Topics

In this scenario, there is only one tenant but you want to control
which agent handles which topic. This is useful when you want to
create silos and have different agents specialized in handling different topics.

To apply type-based subscription for this scenario, 
create one type-based subscription for each agent type but with different
topic types. You can map the same topic type to multiple agent types if
you want these agent types to share a same topic.
For topic source, still use the same value for all messages when you publish.

Continuing the example above with same agent types, create the following
type-based subscriptions:

```python
# Type-based Subscriptions for single-tenant, multiple topics scenario
TypeSubscription(topic_type="triage", agent_type="triage_agent")
TypeSubscription(topic_type="coding", agent_type="coder_agent")
TypeSubscription(topic_type="coding", agent_type="reviewer_agent")
```

With the above type-based subscriptions, any message published to the topic
`("triage", "default")` will be delivered to the agent with type
`"triage_agent"`, and any message published to the topic
`("coding", "default")` will be delivered to the agents with types
`"coder_agent"` and `"reviewer_agent"`. 

The following figure shows how type-based subscription works in this example.

![Type-Based Subscription Single-Tenant, Multiple Topics Scenario Example](type-subscription-single-tenant-multiple-topics.svg)


###### Multi-Tenant Scenarios

In single-tenant scenarios, the topic source is always the same (e.g., `"default"`)
-- it is hard-coded in the application code.
When moving to multi-tenant scenarios, the topic source becomes data-dependent.

```{note}
A good indication that you are in a multi-tenant scenario is that you need 
multiple instances of the same agent type. For example, you may want to have
different agent instances to handle different user sessions to 
keep private data isolated, or, you may want to distribute a heavy workload
across multiple instances of the same agent type and have them work on it concurrently.
```

Continuing the example above, if you want to have dedicated instances of agents
to handle a specific GitHub issue, you need to set the topic source to be a
unique identifier for the issue. 

For example, let's say there is one type-based subscription for the agent type
`"triage_agent"`:

```python
TypeSubscription(topic_type="github_issues", agent_type="triage_agent")
```

When a message is published to the topic
`("github_issues", "github.com/microsoft/autogen/issues/1")`,
the runtime will deliver the message to the agent with ID
`("triage_agent", "github.com/microsoft/autogen/issues/1")`.
When a message is published to the topic
`("github_issues", "github.com/microsoft/autogen/issues/9")`,
the runtime will deliver the message to the agent with ID
`("triage_agent", "github.com/microsoft/autogen/issues/9")`.

The following figure shows how type-based subscription works in this example.

![Type-Based Subscription Multi-Tenant Scenario Example](type-subscription-multi-tenant.svg)

Note the agent ID is data-dependent, and the runtime will create a new instance
of the agent
if it does not exist.

To support multiple topics per tenant, you can use different topic types,
just like the single-tenant, multiple topics scenario.



---

# Framework Guide


## Agent and Agent Runtime

*[framework/agent-and-agent-runtime.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/agent-and-agent-runtime.html)*

### Agent and Agent Runtime

In this and the following section, we focus on the core concepts of AutoGen:
agents, agent runtime, messages, and communication -- 
the foundational building blocks for an multi-agent applications.

```{note}
The Core API is designed to be unopinionated and flexible. So at times, you
may find it challenging. Continue if you are building
an interactive, scalable and distributed multi-agent system and want full control
of all workflows.
If you just want to get something running
quickly, you may take a look at the [AgentChat API](../../agentchat-user-guide/index.md).
```

An agent in AutoGen is an entity defined by the base interface {py:class}`~autogen_core.Agent`.
It has a unique identifier of the type {py:class}`~autogen_core.AgentId`,
a metadata dictionary of the type {py:class}`~autogen_core.AgentMetadata`.

In most cases, you can subclass your agents from higher level class {py:class}`~autogen_core.RoutedAgent` which enables you to route messages to corresponding message handler specified with {py:meth}`~autogen_core.message_handler` decorator and proper type hint for the `message` variable.
An agent runtime is the execution environment for agents in AutoGen.

Similar to the runtime environment of a programming language,
an agent runtime provides the necessary infrastructure to facilitate communication
between agents, manage agent lifecycles, enforce security boundaries, and support monitoring and
debugging.

For local development, developers can use {py:class}`~autogen_core.SingleThreadedAgentRuntime`,
which can be embedded in a Python application.

```{note}
Agents are not directly instantiated and managed by application code.
Instead, they are created by the runtime when needed and managed by the runtime.

If you are already familiar with [AgentChat](../../agentchat-user-guide/index.md),
it is important to note that AgentChat's agents such as
{py:class}`~autogen_agentchat.agents.AssistantAgent` are created by application 
and thus not directly managed by the runtime. To use an AgentChat agent in Core,
you need to create a wrapper Core agent that delegates messages to the AgentChat agent
and let the runtime manage the wrapper agent.
```

#### Implementing an Agent

To implement an agent, the developer must subclass the {py:class}`~autogen_core.RoutedAgent` class
and implement a message handler method for each message type the agent is expected to handle using
the {py:meth}`~autogen_core.message_handler` decorator.
For example,
the following agent handles a simple message type `MyMessageType` and prints the message it receives:

```python
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler


@dataclass
class MyMessageType:
    content: str


class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
```

This agent only handles `MyMessageType` and messages will be delivered to `handle_my_message_type` method. Developers can have multiple message handlers for different message types by using {py:meth}`~autogen_core.message_handler` decorator and setting the type hint for the `message` variable in the handler function. You can also leverage [python typing union](https://docs.python.org/3/library/typing.html#typing.Union) for the `message` variable in one message handler function if it better suits agent's logic.
See the next section on [message and communication](./message-and-communication.ipynb).

#### Using an AgentChat Agent

If you have an [AgentChat](../../agentchat-user-guide/index.md) agent and want to use it in the Core API, you can create
a wrapper {py:class}`~autogen_core.RoutedAgent` that delegates messages to the AgentChat agent.
The following example shows how to create a wrapper agent for the {py:class}`~autogen_agentchat.agents.AssistantAgent`
in AgentChat.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


class MyAssistant(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message}")
```

For how to use model client, see the [Model Client](../components/model-clients.ipynb) section.

Since the Core API is unopinionated,
you are not required to use the AgentChat API to use the Core API.
You can implement your own agents or use another agent framework.

#### Registering Agent Type

To make agents available to the runtime, developers can use the
{py:meth}`~autogen_core.BaseAgent.register` class method of the
{py:class}`~autogen_core.BaseAgent` class.
The process of registration associates an agent type, which is uniquely identified by a string, 
and a factory function
that creates an instance of the agent type of the given class.
The factory function is used to allow automatic creation of agent instances 
when they are needed.

Agent type ({py:class}`~autogen_core.AgentType`) is not the same as the agent class. In this example,
the agent type is `AgentType("my_agent")` or `AgentType("my_assistant")` and the agent class is the Python class `MyAgent` or `MyAssistantAgent`.
The factory function is expected to return an instance of the agent class 
on which the {py:meth}`~autogen_core.BaseAgent.register` class method is invoked.
Read [Agent Identity and Lifecycles](../core-concepts/agent-identity-and-lifecycle.md)
to learn more about agent type and identity.

```{note}
Different agent types can be registered with factory functions that return 
the same agent class. For example, in the factory functions, 
variations of the constructor parameters
can be used to create different instances of the same agent class.
```

To register our agent types with the 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
the following code can be used:

```python
from autogen_core import SingleThreadedAgentRuntime

runtime = SingleThreadedAgentRuntime()
await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
await MyAssistant.register(runtime, "my_assistant", lambda: MyAssistant("my_assistant"))
```

```text
AgentType(type='my_assistant')
```

Once an agent type is registered, we can send a direct message to an agent instance
using an {py:class}`~autogen_core.AgentId`.
The runtime will create the instance the first time it delivers a
message to this instance.

```python
runtime.start()  # Start processing messages in the background.
await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_assistant", "default"))
await runtime.stop()  # Stop processing messages in the background.
```

```text
my_agent received message: Hello, World!
my_assistant received message: Hello, World!
my_assistant responded: Hello! How can I assist you today?
```

```{note}
Because the runtime manages the lifecycle of agents, an {py:class}`~autogen_core.AgentId`
is only used to communicate with the agent or retrieve its metadata (e.g., description).
```

#### Running the Single-Threaded Agent Runtime

The above code snippet uses {py:meth}`~autogen_core.SingleThreadedAgentRuntime.start` to start a background task
to process and deliver messages to recepients' message handlers.
This is a feature of the
local embedded runtime {py:class}`~autogen_core.SingleThreadedAgentRuntime`.

To stop the background task immediately, use the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.stop` method:

```python
runtime.start()
# ... Send messages, publish messages, etc.
await runtime.stop()  # This will return immediately but will not cancel
# any in-progress message handling.
```

You can resume the background task by calling {py:meth}`~autogen_core.SingleThreadedAgentRuntime.start` again.

For batch scenarios such as running benchmarks for evaluating agents,
you may want to wait for the background task to stop automatically when
there are no unprocessed messages and no agent is handling messages --
the batch may considered complete.
You can achieve this by using the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.stop_when_idle` method:

```python
runtime.start()
# ... Send messages, publish messages, etc.
await runtime.stop_when_idle()  # This will block until the runtime is idle.
```

To close the runtime and release resources, use the {py:meth}`~autogen_core.SingleThreadedAgentRuntime.close` method:

```python
await runtime.close()
```

Other runtime implementations will have their own ways of running the runtime.


## Message and Communication

*[framework/message-and-communication.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html)*

### Message and Communication

An agent in AutoGen core can react to, send, and publish messages,
and messages are the only means through which agents can communicate
with each other.

#### Messages

Messages are serializable objects, they can be defined using:

- A subclass of Pydantic's {py:class}`pydantic.BaseModel`, or
- A dataclass

For example:

```python
from dataclasses import dataclass


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class ImageMessage:
    url: str
    source: str
```

```{note}
Messages are purely data, and should not contain any logic.
```

#### Message Handlers

When an agent receives a message the runtime will invoke the agent's message handler
({py:meth}`~autogen_core.Agent.on_message`) which should implement the agents message handling logic.
If this message cannot be handled by the agent, the agent should raise a
{py:class}`~autogen_core.exceptions.CantHandleException`.

The base class {py:class}`~autogen_core.BaseAgent` provides no message handling logic
and implementing the {py:meth}`~autogen_core.Agent.on_message` method directly is not recommended
unless for the advanced use cases.

Developers should start with implementing the {py:class}`~autogen_core.RoutedAgent` base class
which provides built-in message routing capability.

##### Routing Messages by Type

The {py:class}`~autogen_core.RoutedAgent` base class provides a mechanism
for associating message types with message handlers 
with the {py:meth}`~autogen_core.components.message_handler` decorator,
so developers do not need to implement the {py:meth}`~autogen_core.Agent.on_message` method.

For example, the following type-routed agent responds to `TextMessage` and `ImageMessage`
using different message handlers:

```python
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler


class MyAgent(RoutedAgent):
    @message_handler
    async def on_text_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you said {message.content}!")

    @message_handler
    async def on_image_message(self, message: ImageMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you sent me {message.url}!")
```

Create the agent runtime and register the agent type (see [Agent and Agent Runtime](agent-and-agent-runtime.ipynb)):

```python
runtime = SingleThreadedAgentRuntime()
await MyAgent.register(runtime, "my_agent", lambda: MyAgent("My Agent"))
```

```text
AgentType(type='my_agent')
```

Test this agent with `TextMessage` and `ImageMessage`.

```python
runtime.start()
agent_id = AgentId("my_agent", "default")
await runtime.send_message(TextMessage(content="Hello, World!", source="User"), agent_id)
await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="User"), agent_id)
await runtime.stop_when_idle()
```

```text
Hello, User, you said Hello, World!!
Hello, User, you sent me https://example.com/image.jpg!
```

The runtime automatically creates an instance of `MyAgent` with the 
agent ID `AgentId("my_agent", "default")` when delivering the first message.

##### Routing Messages of the Same Type

In some scenarios, it is useful to route messages of the same type to different handlers.
For examples, messages from different sender agents should be handled differently.
You can use the `match` parameter of the {py:meth}`~autogen_core.components.message_handler` decorator.

The `match` parameter associates handlers for the same message type
to a specific message -- it is secondary to the message type routing. 
It accepts a callable that takes the message and 
{py:class}`~autogen_core.MessageContext` as arguments, and
returns a boolean indicating whether the message should be handled by the decorated handler.
The callable is checked in the alphabetical order of the handlers.

Here is an example of an agent that routes messages based on the sender agent
using the `match` parameter:

```python
class RoutedBySenderAgent(RoutedAgent):
    @message_handler(match=lambda msg, ctx: msg.source.startswith("user1"))  # type: ignore
    async def on_user1_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello from user 1 handler, {message.source}, you said {message.content}!")

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_user2_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello from user 2 handler, {message.source}, you said {message.content}!")

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_image_message(self, message: ImageMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you sent me {message.url}!")
```

The above agent uses the `source` field of the message to determine the sender agent.
You can also use the `sender` field of {py:class}`~autogen_core.MessageContext` to determine the sender agent
using the agent ID if available.

Let's test this agent with messages with different `source` values:

```python
runtime = SingleThreadedAgentRuntime()
await RoutedBySenderAgent.register(runtime, "my_agent", lambda: RoutedBySenderAgent("Routed by sender agent"))
runtime.start()
agent_id = AgentId("my_agent", "default")
await runtime.send_message(TextMessage(content="Hello, World!", source="user1-test"), agent_id)
await runtime.send_message(TextMessage(content="Hello, World!", source="user2-test"), agent_id)
await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="user1-test"), agent_id)
await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="user2-test"), agent_id)
await runtime.stop_when_idle()
```

```text
Hello from user 1 handler, user1-test, you said Hello, World!!
Hello from user 2 handler, user2-test, you said Hello, World!!
Hello, user2-test, you sent me https://example.com/image.jpg!
```

In the above example, the first `ImageMessage` is not handled because the `source` field
of the message does not match the handler's `match` condition.

#### Direct Messaging

There are two types of communication in AutoGen core:

- **Direct Messaging**: sends a direct message to another agent.
- **Broadcast**: publishes a message to a topic.

Let's first look at direct messaging.
To send a direct message to another agent, within a message handler use
the {py:meth}`autogen_core.BaseAgent.send_message` method,
from the runtime use the {py:meth}`autogen_core.AgentRuntime.send_message` method.
Awaiting calls to these methods will return the return value of the
receiving agent's message handler.
When the receiving agent's handler returns `None`, `None` will be returned.

```{note}
If the invoked agent raises an exception while the sender is awaiting,
the exception will be propagated back to the sender.
```

##### Request/Response

Direct messaging can be used for request/response scenarios,
where the sender expects a response from the receiver.
The receiver can respond to the message by returning a value from its message handler.
You can think of this as a function call between agents.

For example, consider the following agents:

```python
from dataclasses import dataclass

from autogen_core import MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler


@dataclass
class Message:
    content: str


class InnerAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> Message:
        return Message(content=f"Hello from inner, {message.content}")


class OuterAgent(RoutedAgent):
    def __init__(self, description: str, inner_agent_type: str):
        super().__init__(description)
        self.inner_agent_id = AgentId(inner_agent_type, self.id.key)

    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"Received message: {message.content}")
        # Send a direct message to the inner agent and receives a response.
        response = await self.send_message(Message(f"Hello from outer, {message.content}"), self.inner_agent_id)
        print(f"Received inner response: {response.content}")
```

Upone receving a message, the `OuterAgent` sends a direct message to the `InnerAgent` and receives
a message in response.

We can test these agents by sending a `Message` to the `OuterAgent`.

```python
runtime = SingleThreadedAgentRuntime()
await InnerAgent.register(runtime, "inner_agent", lambda: InnerAgent("InnerAgent"))
await OuterAgent.register(runtime, "outer_agent", lambda: OuterAgent("OuterAgent", "inner_agent"))
runtime.start()
outer_agent_id = AgentId("outer_agent", "default")
await runtime.send_message(Message(content="Hello, World!"), outer_agent_id)
await runtime.stop_when_idle()
```

```text
Received message: Hello, World!
Received inner response: Hello from inner, Hello from outer, Hello, World!
```

Both outputs are produced by the `OuterAgent`'s message handler, however the second output is based on the response from the `InnerAgent`.

Generally speaking, direct messaging is appropriate for scenarios when the sender and
recipient are tightly coupled -- they are created together and the sender
is linked to a specific instance of the recipient.
For example, an agent executes tool calls by sending direct messages to
an instance of {py:class}`~autogen_core.tool_agent.ToolAgent`,
and uses the responses to form an action-observation loop.

#### Broadcast

Broadcast is effectively the publish/subscribe model with topic and subscription.
Read [Topic and Subscription](../core-concepts/topic-and-subscription.md)
to learn the core concepts.

The key difference between direct messaging and broadcast is that broadcast
cannot be used for request/response scenarios.
When an agent publishes a message it is one way only, it cannot receive a response
from any other agent, even if a receiving agent's handler returns a value.

```{note}
If a response is given to a published message, it will be thrown away.
```

```{note}
If an agent publishes a message type for which it is subscribed it will not
receive the message it published. This is to prevent infinite loops.
```

##### Subscribe and Publish to Topics

[Type-based subscription](../core-concepts/topic-and-subscription.md#type-based-subscription)
maps messages published to topics of a given topic type to 
agents of a given agent type. 
To make an agent that subsclasses {py:class}`~autogen_core.RoutedAgent`
subscribe to a topic of a given topic type,
you can use the {py:meth}`~autogen_core.components.type_subscription` class decorator.

The following example shows a `ReceiverAgent` class that subscribes to topics of `"default"` topic type
using the {py:meth}`~autogen_core.components.type_subscription` decorator.
and prints the received messages.

```python
from autogen_core import RoutedAgent, message_handler, type_subscription


@type_subscription(topic_type="default")
class ReceivingAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"Received a message: {message.content}")
```

To publish a message from an agent's handler,
use the {py:meth}`~autogen_core.BaseAgent.publish_message` method and specify
a {py:class}`~autogen_core.TopicId`.
This call must still be awaited to allow the runtime to schedule delivery of 
the message to all subscribers, but it will always return `None`.
If an agent raises an exception while handling a published message,
this will be logged but will not be propagated back to the publishing agent.

The following example shows a `BroadcastingAgent` that 
publishes a message to a topic upon receiving a message.

```python
from autogen_core import TopicId


class BroadcastingAgent(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        await self.publish_message(
            Message("Publishing a message from broadcasting agent!"),
            topic_id=TopicId(type="default", source=self.id.key),
        )
```

`BroadcastingAgent` publishes message to a topic with type `"default"`
and source assigned to the agent instance's agent key.

Subscriptions are registered with the agent runtime, either as part of
agent type's registration or through a separate API method.
Here is how we register {py:class}`~autogen_core.components.TypeSubscription`
for the receiving agent with the {py:meth}`~autogen_core.components.type_subscription` decorator,
and for the broadcasting agent without the decorator.

```python
from autogen_core import TypeSubscription

runtime = SingleThreadedAgentRuntime()

# Option 1: with type_subscription decorator
# The type_subscription class decorator automatically adds a TypeSubscription to
# the runtime when the agent is registered.
await ReceivingAgent.register(runtime, "receiving_agent", lambda: ReceivingAgent("Receiving Agent"))

# Option 2: with TypeSubscription
await BroadcastingAgent.register(runtime, "broadcasting_agent", lambda: BroadcastingAgent("Broadcasting Agent"))
await runtime.add_subscription(TypeSubscription(topic_type="default", agent_type="broadcasting_agent"))

# Start the runtime and publish a message.
runtime.start()
await runtime.publish_message(
    Message("Hello, World! From the runtime!"), topic_id=TopicId(type="default", source="default")
)
await runtime.stop_when_idle()
```

```text
Received a message: Hello, World! From the runtime!
Received a message: Publishing a message from broadcasting agent!
```

As shown in the above example, you can also publish directly to a topic
through the runtime's {py:meth}`~autogen_core.AgentRuntime.publish_message` method
without the need to create an agent instance.

From the output, you can see two messages were received by the receiving agent:
one was published through the runtime, and the other was published by the broadcasting agent.

##### Default Topic and Subscriptions

In the above example, we used
{py:class}`~autogen_core.TopicId` and {py:class}`~autogen_core.components.TypeSubscription`
to specify the topic and subscriptions respectively.
This is the appropriate way for many scenarios.
However, when there is a single scope of publishing, that is, 
all agents publish and subscribe to all broadcasted messages,
we can use the convenience classes {py:class}`~autogen_core.components.DefaultTopicId`
and {py:meth}`~autogen_core.components.default_subscription` to simplify our code.

{py:class}`~autogen_core.components.DefaultTopicId` is
for creating a topic that uses `"default"` as the default value for the topic type
and the publishing agent's key as the default value for the topic source.
{py:meth}`~autogen_core.components.default_subscription` is
for creating a type subscription that subscribes to the default topic.
We can simplify `BroadcastingAgent` by using
{py:class}`~autogen_core.components.DefaultTopicId` and {py:meth}`~autogen_core.components.default_subscription`.

```python
from autogen_core import DefaultTopicId, default_subscription


@default_subscription
class BroadcastingAgentDefaultTopic(RoutedAgent):
    @message_handler
    async def on_my_message(self, message: Message, ctx: MessageContext) -> None:
        # Publish a message to all agents in the same namespace.
        await self.publish_message(
            Message("Publishing a message from broadcasting agent!"),
            topic_id=DefaultTopicId(),
        )
```

When the runtime calls {py:meth}`~autogen_core.BaseAgent.register` to register the agent type,
it creates a {py:class}`~autogen_core.components.TypeSubscription`
whose topic type uses `"default"` as the default value and 
agent type uses the same agent type that is being registered in the same context.

```python
runtime = SingleThreadedAgentRuntime()
await BroadcastingAgentDefaultTopic.register(
    runtime, "broadcasting_agent", lambda: BroadcastingAgentDefaultTopic("Broadcasting Agent")
)
await ReceivingAgent.register(runtime, "receiving_agent", lambda: ReceivingAgent("Receiving Agent"))
runtime.start()
await runtime.publish_message(Message("Hello, World! From the runtime!"), topic_id=DefaultTopicId())
await runtime.stop_when_idle()
```

```text
Received a message: Hello, World! From the runtime!
Received a message: Publishing a message from broadcasting agent!
```

```{note}
If your scenario allows all agents to publish and subscribe to
all broadcasted messages, use {py:class}`~autogen_core.components.DefaultTopicId`
and {py:meth}`~autogen_core.components.default_subscription` to decorate your
agent classes.
```


## Logging

*[framework/logging.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/logging.html)*

### Logging

AutoGen uses Python's built-in [`logging`](https://docs.python.org/3/library/logging.html) module.

There are two kinds of logging:

- **Trace logging**: This is used for debugging and is human readable messages to indicate what is going on. This is intended for a developer to understand what is happening in the code. The content and format of these logs should not be depended on by other systems.
  - Name: {py:attr}`~autogen_core.TRACE_LOGGER_NAME`.
- **Structured logging**: This logger emits structured events that can be consumed by other systems. The content and format of these logs can be depended on by other systems.
  - Name: {py:attr}`~autogen_core.EVENT_LOGGER_NAME`.
  - See the module {py:mod}`autogen_core.logging` to see the available events.
- {py:attr}`~autogen_core.ROOT_LOGGER_NAME` can be used to enable or disable all logs.

#### Enabling logging output

To enable trace logging, you can use the following code:

```python
import logging

from autogen_core import TRACE_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
```

To enable structured logging, you can use the following code:

```python
import logging

from autogen_core import EVENT_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
```

##### Structured logging

Structured logging allows you to write handling logic that deals with the actual events including all fields rather than just a formatted string.

For example, if you had defined this custom event and were emitting it. Then you could write the following handler to receive it.

```python
import logging
from dataclasses import dataclass

@dataclass
class MyEvent:
    timestamp: str
    message: str

class MyHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, MyEvent):
                print(f"Timestamp: {record.msg.timestamp}, Message: {record.msg.message}")
        except Exception:
            self.handleError(record)
```

And this is how you could use it:

```python
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.INFO)
my_handler = MyHandler()
logger.handlers = [my_handler]
```

#### Emitting logs

These two names are the root loggers for these types. Code that emits logs should use a child logger of these loggers. For example, if you are writing a module `my_module` and you want to emit trace logs, you should use the logger named:

```python
import logging

from autogen_core import TRACE_LOGGER_NAME
logger = logging.getLogger(f"{TRACE_LOGGER_NAME}.my_module")
```

##### Emitting structured logs

If your event is a dataclass, then it could be emitted in code like this:

```python
import logging
from dataclasses import dataclass
from autogen_core import EVENT_LOGGER_NAME

@dataclass
class MyEvent:
    timestamp: str
    message: str

logger = logging.getLogger(EVENT_LOGGER_NAME + ".my_module")
logger.info(MyEvent("timestamp", "message"))
```



## Open Telemetry

*[framework/telemetry.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/telemetry.html)*

### Open Telemetry

AutoGen has native support for [open telemetry](https://opentelemetry.io/). This allows you to collect telemetry data from your application and send it to a telemetry backend of your choosing.

These are the components that are currently instrumented:

- Runtime ({py:class}`~autogen_core.SingleThreadedAgentRuntime` and {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime`).
- Tool ({py:class}`~autogen_core.tools.BaseTool`) with the `execute_tool` span in [GenAI semantic convention for tools](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span).
- AgentChat Agents ({py:class}`~autogen_agentchat.agents.BaseChatAgent`) with the `create_agent` and `invoke_agent` spans in [GenAI semantic convention for agents](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#create-agent-span).

```{note}
To disable the agent runtime telemetry, you can set the `trace_provider` to
`opentelemetry.trace.NoOpTracerProvider` in the runtime constructor.

Additionally, you can set the environment variable `AUTOGEN_DISABLE_RUNTIME_TRACING` to `true` to disable the agent runtime telemetry if you don't have access to the runtime constructor. For example, if you are using `ComponentConfig`.
```

#### Instrumenting your application

To instrument your application, you will need an sdk and an exporter. You may already have these if your application is already instrumented with open telemetry.

#### Clean instrumentation

If you do not have open telemetry set up in your application, you can follow these steps to instrument your application.

```bash
pip install opentelemetry-sdk
```

Depending on your open telemetry collector, you can use grpc or http to export your telemetry.

```bash
# Pick one of the following

pip install opentelemetry-exporter-otlp-proto-http
pip install opentelemetry-exporter-otlp-proto-grpc
```

Next, we need to get a tracer provider:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def configure_oltp_tracing(endpoint: str = None) -> trace.TracerProvider:
    # Configure Tracing
    tracer_provider = TracerProvider(resource=Resource({"service.name": "my-service"}))
    processor = BatchSpanProcessor(OTLPSpanExporter())
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider
```

Now you can send the trace_provider when creating your runtime:

```python
# for single threaded runtime
single_threaded_runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
# or for worker runtime
worker_runtime = GrpcWorkerAgentRuntime(tracer_provider=tracer_provider)
```

And that's it! Your application is now instrumented with open telemetry. You can now view your telemetry data in your telemetry backend.

##### Existing instrumentation

If you have open telemetry already set up in your application, you can pass the tracer provider to the runtime when creating it:

```python
from opentelemetry import trace

# Get the tracer provider from your application
tracer_provider = trace.get_tracer_provider()

# for single threaded runtime
single_threaded_runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
# or for worker runtime
worker_runtime = GrpcWorkerAgentRuntime(tracer_provider=tracer_provider)
```

##### Examples

See [Tracing and Observability](../../agentchat-user-guide/tracing.ipynb)
for a complete example of how to set up open telemetry with AutoGen.



## Distributed Agent Runtime

*[framework/distributed-agent-runtime.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/distributed-agent-runtime.html)*

### Distributed Agent Runtime

```{attention}
The distributed agent runtime is an experimental feature. Expect breaking changes
to the API.
```

A distributed agent runtime facilitates communication and agent lifecycle management
across process boundaries.
It consists of a host service and at least one worker runtime.

The host service maintains connections to all active worker runtimes,
facilitates message delivery, and keeps sessions for all direct messages (i.e., RPCs).
A worker runtime processes application code (agents) and connects to the host service.
It also advertises the agents which they support to the host service,
so the host service can deliver messages to the correct worker.

````{note}
The distributed agent runtime requires extra dependencies, install them using:
```bash
pip install "autogen-ext[grpc]"
```
````

We can start a host service using {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntimeHost`.

```python
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

host = GrpcWorkerAgentRuntimeHost(address="localhost:50051")
host.start()  # Start a host service in the background.
```

The above code starts the host service in the background and accepts
worker connections on port 50051.

Before running worker runtimes, let's define our agent.
The agent will publish a new message on every message it receives.
It also keeps track of how many messages it has published, and 
stops publishing new messages once it has published 5 messages.

```python
from dataclasses import dataclass

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler


@dataclass
class MyMessage:
    content: str


@default_subscription
class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__("My agent")
        self._name = name
        self._counter = 0

    @message_handler
    async def my_message_handler(self, message: MyMessage, ctx: MessageContext) -> None:
        self._counter += 1
        if self._counter > 5:
            return
        content = f"{self._name}: Hello x {self._counter}"
        print(content)
        await self.publish_message(MyMessage(content=content), DefaultTopicId())
```

Now we can set up the worker agent runtimes.
We use {py:class}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime`.
We set up two worker runtimes. Each runtime hosts one agent.
All agents publish and subscribe to the default topic, so they can see all
messages being published.

To run the agents, we publish a message from a worker.

```python
import asyncio

from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime

worker1 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
await worker1.start()
await MyAgent.register(worker1, "worker1", lambda: MyAgent("worker1"))

worker2 = GrpcWorkerAgentRuntime(host_address="localhost:50051")
await worker2.start()
await MyAgent.register(worker2, "worker2", lambda: MyAgent("worker2"))

await worker2.publish_message(MyMessage(content="Hello!"), DefaultTopicId())

# Let the agents run for a while.
await asyncio.sleep(5)
```

```text
worker1: Hello x 1
worker2: Hello x 1
worker2: Hello x 2
worker1: Hello x 2
worker1: Hello x 3
worker2: Hello x 3
worker2: Hello x 4
worker1: Hello x 4
worker1: Hello x 5
worker2: Hello x 5
```

We can see each agent published exactly 5 messages.

To stop the worker runtimes, we can call {py:meth}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntime.stop`.

```python
await worker1.stop()
await worker2.stop()

# To keep the worker running until a termination signal is received (e.g., SIGTERM).
# await worker1.stop_when_signal()
```

We can call {py:meth}`~autogen_ext.runtimes.grpc.GrpcWorkerAgentRuntimeHost.stop`
to stop the host service.

```python
await host.stop()

# To keep the host service running until a termination signal (e.g., SIGTERM)
# await host.stop_when_signal()
```

#### Cross-Language Runtimes
The process described above is largely the same, however all message types MUST use shared protobuf schemas for all cross-agent message types.

#### Next Steps
To see complete examples of using distributed runtime, please take a look at the following samples:

- [Distributed Workers](https://github.com/microsoft/autogen/tree/main/python/samples/core_grpc_worker_runtime)  
- [Distributed Semantic Router](https://github.com/microsoft/autogen/tree/main/python/samples/core_semantic_router)  
- [Distributed Group Chat](https://github.com/microsoft/autogen/tree/main/python/samples/core_distributed-group-chat)


## Component config

*[framework/component-config.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/component-config.html)*

### Component config

AutoGen components are able to be declaratively configured in a generic fashion. This is to support configuration based experiences, such as AutoGen studio, but it is also useful for many other scenarios.

The system that provides this is called "component configuration". In AutoGen, a component is simply something that can be created from a config object and itself can be dumped to a config object. In this way, you can define a component in code and then get the config object from it.

This system is generic and allows for components defined outside of AutoGen itself (such as extensions) to be configured in the same way.

#### How does this differ from state?

This is a very important point to clarify. When we talk about serializing an object, we must include *all* data that makes that object itself. Including things like message history etc. When deserializing from serialized state, you must get back the *exact* same object. This is not the case with component configuration.

Component configuration should be thought of as the blueprint for an object, and can be stamped out many times to create many instances of the same configured object.

#### Usage

If you have a component in Python and want to get the config for it, simply call {py:meth}`~autogen_core.ComponentToConfig.dump_component` on it. The resulting object can be passed back into {py:meth}`~autogen_core.ComponentLoader.load_component` to get the component back.

##### Loading a component from a config

To load a component from a config object, you can use the {py:meth}`~autogen_core.ComponentLoader.load_component` method. This method will take a config object and return a component object. It is best to call this method on the interface you want. For example to load a model client:

```python
from autogen_core.models import ChatCompletionClient

config = {
    "provider": "openai_chat_completion_client",
    "config": {"model": "gpt-4o"},
}

client = ChatCompletionClient.load_component(config)
```

#### Creating a component class

To add component functionality to a given class:

1. Add a call to {py:meth}`~autogen_core.Component` in the class inheritance list.
2. Implment the {py:meth}`~autogen_core.ComponentToConfig._to_config` and {py:meth}`~autogen_core.ComponentFromConfig._from_config` methods

For example:

```python
from autogen_core import Component, ComponentBase
from pydantic import BaseModel


class Config(BaseModel):
    value: str


class MyComponent(ComponentBase[Config], Component[Config]):
    component_type = "custom"
    component_config_schema = Config

    def __init__(self, value: str):
        self.value = value

    def _to_config(self) -> Config:
        return Config(value=self.value)

    @classmethod
    def _from_config(cls, config: Config) -> "MyComponent":
        return cls(value=config.value)
```


#### Secrets

If a field of a config object is a secret value, it should be marked using [`SecretStr`](https://docs.pydantic.dev/latest/api/types/#pydantic.types.SecretStr), this will ensure that the value will not be dumped to the config object.

For example:

```python
from pydantic import BaseModel, SecretStr


class ClientConfig(BaseModel):
    endpoint: str
    api_key: SecretStr
```


---

# Components Guide


## Model Clients

*[components/model-clients.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-clients.html)*

### Model Clients

AutoGen provides a suite of built-in model clients for using ChatCompletion API.
All model clients implement the {py:class}`~autogen_core.models.ChatCompletionClient` protocol class.

Currently we support the following built-in model clients:
* {py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient`: for OpenAI models and models with OpenAI API compatibility (e.g., Gemini).
* {py:class}`~autogen_ext.models.openai.AzureOpenAIChatCompletionClient`: for Azure OpenAI models.
* {py:class}`~autogen_ext.models.azure.AzureAIChatCompletionClient`: for GitHub models and models hosted on Azure.
* {py:class}`~autogen_ext.models.ollama.OllamaChatCompletionClient` (Experimental): for local models hosted on Ollama.
* {py:class}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient` (Experimental): for models hosted on Anthropic.
* {py:class}`~autogen_ext.models.semantic_kernel.SKChatCompletionAdapter`: adapter for Semantic Kernel AI connectors.

For more information on how to use these model clients, please refer to the documentation of each client.

#### Log Model Calls

AutoGen uses standard Python logging module to log events like model calls and responses.
The logger name is {py:attr}`autogen_core.EVENT_LOGGER_NAME`, and the event type is `LLMCall`.

```python
import logging

from autogen_core import EVENT_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
```

#### Call Model Client

To call a model client, you can use the {py:meth}`~autogen_core.models.ChatCompletionClient.create` method.
This example uses the {py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient` to call an OpenAI model.

```python
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4", temperature=0.3
)  # assuming OPENAI_API_KEY is set in the environment.

result = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
print(result)
```

```text
finish_reason='stop' content='The capital of France is Paris.' usage=RequestUsage(prompt_tokens=15, completion_tokens=8) cached=False logprobs=None thought=None
```

#### Streaming Tokens

You can use the {py:meth}`~autogen_core.models.ChatCompletionClient.create_stream` method to create a
chat completion request with streaming token chunks.

```python
from autogen_core.models import CreateResult, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")  # assuming OPENAI_API_KEY is set in the environment.

messages = [
    UserMessage(content="Write a very short story about a dragon.", source="user"),
]

# Create a stream.
stream = model_client.create_stream(messages=messages)

# Iterate over the stream and print the responses.
print("Streamed responses:")
async for chunk in stream:  # type: ignore
    if isinstance(chunk, str):
        # The chunk is a string.
        print(chunk, flush=True, end="")
    else:
        # The final chunk is a CreateResult object.
        assert isinstance(chunk, CreateResult) and isinstance(chunk.content, str)
        # The last response is a CreateResult object with the complete message.
        print("\n\n------------\n")
        print("The complete response:", flush=True)
        print(chunk.content, flush=True)
```

```text
Streamed responses:
In the heart of an ancient forest, beneath the shadow of snow-capped peaks, a dragon named Elara lived secretly for centuries. Elara was unlike any dragon from the old tales; her scales shimmered with a deep emerald hue, each scale engraved with symbols of lost wisdom. The villagers in the nearby valley spoke of mysterious lights dancing across the night sky, but none dared venture close enough to solve the enigma.

One cold winter's eve, a young girl named Lira, brimming with curiosity and armed with the innocence of youth, wandered into Elara’s domain. Instead of fire and fury, she found warmth and a gentle gaze. The dragon shared stories of a world long forgotten and in return, Lira gifted her simple stories of human life, rich in laughter and scent of earth.

From that night on, the villagers noticed subtle changes—the crops grew taller, and the air seemed sweeter. Elara had infused the valley with ancient magic, a guardian of balance, watching quietly as her new friend thrived under the stars. And so, Lira and Elara’s bond marked the beginning of a timeless friendship that spun tales of hope whispered through the leaves of the ever-verdant forest.

------------

The complete response:
In the heart of an ancient forest, beneath the shadow of snow-capped peaks, a dragon named Elara lived secretly for centuries. Elara was unlike any dragon from the old tales; her scales shimmered with a deep emerald hue, each scale engraved with symbols of lost wisdom. The villagers in the nearby valley spoke of mysterious lights dancing across the night sky, but none dared venture close enough to solve the enigma.

One cold winter's eve, a young girl named Lira, brimming with curiosity and armed with the innocence of youth, wandered into Elara’s domain. Instead of fire and fury, she found warmth and a gentle gaze. The dragon shared stories of a world long forgotten and in return, Lira gifted her simple stories of human life, rich in laughter and scent of earth.

From that night on, the villagers noticed subtle changes—the crops grew taller, and the air seemed sweeter. Elara had infused the valley with ancient magic, a guardian of balance, watching quietly as her new friend thrived under the stars. And so, Lira and Elara’s bond marked the beginning of a timeless friendship that spun tales of hope whispered through the leaves of the ever-verdant forest.


------------

The token usage was:
RequestUsage(prompt_tokens=0, completion_tokens=0)
```

```{note}
The last response in the streaming response is always the final response
of the type {py:class}`~autogen_core.models.CreateResult`.
```

```{note}
The default usage response is to return zero values. To enable usage, 
see {py:meth}`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create_stream`
for more details.
```

#### Structured Output

Structured output can be enabled by setting the `response_format` field in
{py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient` and {py:class}`~autogen_ext.models.openai.AzureOpenAIChatCompletionClient` to
as a [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/) class.

```{note}
Structured output is only available for models that support it. It also
requires the model client to support structured output as well.
Currently, the {py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient`
and {py:class}`~autogen_ext.models.openai.AzureOpenAIChatCompletionClient`
support structured output.
```

```python
from typing import Literal

from pydantic import BaseModel


# The response format for the agent as a Pydantic base model.
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


# Create an agent that uses the OpenAI GPT-4o model with the custom response format.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=AgentResponse,  # type: ignore
)

# Send a message list to the model and await the response.
messages = [
    UserMessage(content="I am happy.", source="user"),
]
response = await model_client.create(messages=messages)
assert isinstance(response.content, str)
parsed_response = AgentResponse.model_validate_json(response.content)
print(parsed_response.thoughts)
print(parsed_response.response)

# Close the connection to the model client.
await model_client.close()
```

```text
I'm glad to hear that you're feeling happy! It's such a great emotion that can brighten your whole day. Is there anything in particular that's bringing you joy today? 😊
happy
```

You also use the `extra_create_args` parameter in the {py:meth}`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create` method
to set the `response_format` field so that the structured output can be configured for each request.

#### Caching Model Responses

`autogen_ext` implements {py:class}`~autogen_ext.models.cache.ChatCompletionCache` that can wrap any {py:class}`~autogen_core.models.ChatCompletionClient`. Using this wrapper avoids incurring token usage when querying the underlying client with the same prompt multiple times.

{py:class}`~autogen_core.models.ChatCompletionCache` uses a {py:class}`~autogen_core.CacheStore` protocol. We have implemented some useful variants of {py:class}`~autogen_core.CacheStore` including {py:class}`~autogen_ext.cache_store.diskcache.DiskCacheStore` and {py:class}`~autogen_ext.cache_store.redis.RedisStore`.

Here's an example of using `diskcache` for local caching:

```python
# pip install -U "autogen-ext[openai, diskcache]"
```

```python
import asyncio
import tempfile

from autogen_core.models import UserMessage
from autogen_ext.cache_store.diskcache import DiskCacheStore
from autogen_ext.models.cache import CHAT_CACHE_VALUE_TYPE, ChatCompletionCache
from autogen_ext.models.openai import OpenAIChatCompletionClient
from diskcache import Cache


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize the original client
        openai_model_client = OpenAIChatCompletionClient(model="gpt-4o")

        # Then initialize the CacheStore, in this case with diskcache.Cache.
        # You can also use redis like:
        # from autogen_ext.cache_store.redis import RedisStore
        # import redis
        # redis_instance = redis.Redis()
        # cache_store = RedisCacheStore[CHAT_CACHE_VALUE_TYPE](redis_instance)
        cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(tmpdirname))
        cache_client = ChatCompletionCache(openai_model_client, cache_store)

        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)  # Should print response from OpenAI
        response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
        print(response)  # Should print cached response

        await openai_model_client.close()
        await cache_client.close()


asyncio.run(main())
```

```text
True
```

Inspecting `cached_client.total_usage()` (or `model_client.total_usage()`) before and after a cached response should yield idential counts.

Note that the caching is sensitive to the exact arguments provided to `cached_client.create` or `cached_client.create_stream`, so changing `tools` or `json_output` arguments might lead to a cache miss.

#### Build an Agent with a Model Client

Let's create a simple AI agent that can respond to messages using the ChatCompletion API.

```python
from dataclasses import dataclass

from autogen_core import MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class Message:
    content: str


class SimpleAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        response = await self._model_client.create(
            self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
        )
        # Return with the model's response.
        assert isinstance(response.content, str)
        return Message(content=response.content)
```

The `SimpleAgent` class is a subclass of the
{py:class}`autogen_core.RoutedAgent` class for the convenience of automatically routing messages to the appropriate handlers.
It has a single handler, `handle_user_message`, which handles message from the user. It uses the `ChatCompletionClient` to generate a response to the message.
It then returns the response to the user, following the direct communication model.

```{note}
The `cancellation_token` of the type {py:class}`autogen_core.CancellationToken` is used to cancel
asynchronous operations. It is linked to async calls inside the message handlers
and can be used by the caller to cancel the handlers.
```

```python
# Create the runtime and register the agent.
from autogen_core import AgentId

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY set in the environment.
)

runtime = SingleThreadedAgentRuntime()
await SimpleAgent.register(
    runtime,
    "simple_agent",
    lambda: SimpleAgent(model_client=model_client),
)
# Start the runtime processing messages.
runtime.start()
# Send a message to the agent and get the response.
message = Message("Hello, what are some fun things to do in Seattle?")
response = await runtime.send_message(message, AgentId("simple_agent", "default"))
print(response.content)
# Stop the runtime processing messages.
await runtime.stop()
await model_client.close()
```

```text
Seattle is a vibrant city with a wide range of activities and attractions. Here are some fun things to do in Seattle:

1. **Space Needle**: Visit this iconic observation tower for stunning views of the city and surrounding mountains.

2. **Pike Place Market**: Explore this historic market where you can see the famous fish toss, buy local produce, and find unique crafts and eateries.

3. **Museum of Pop Culture (MoPOP)**: Dive into the world of contemporary culture, music, and science fiction at this interactive museum.

4. **Chihuly Garden and Glass**: Marvel at the beautiful glass art installations by artist Dale Chihuly, located right next to the Space Needle.

5. **Seattle Aquarium**: Discover the diverse marine life of the Pacific Northwest at this engaging aquarium.

6. **Seattle Art Museum**: Explore a vast collection of art from around the world, including contemporary and indigenous art.

7. **Kerry Park**: For one of the best views of the Seattle skyline, head to this small park on Queen Anne Hill.

8. **Ballard Locks**: Watch boats pass through the locks and observe the salmon ladder to see salmon migrating.

9. **Ferry to Bainbridge Island**: Take a scenic ferry ride across Puget Sound to enjoy charming shops, restaurants, and beautiful natural scenery.

10. **Olympic Sculpture Park**: Stroll through this outdoor park with large-scale sculptures and stunning views of the waterfront and mountains.

11. **Underground Tour**: Discover Seattle's history on this quirky tour of the city's underground passageways in Pioneer Square.

12. **Seattle Waterfront**: Enjoy the shops, restaurants, and attractions along the waterfront, including the Seattle Great Wheel and the aquarium.

13. **Discovery Park**: Explore the largest green space in Seattle, featuring trails, beaches, and views of Puget Sound.

14. **Food Tours**: Try out Seattle’s diverse culinary scene, including fresh seafood, international cuisines, and coffee culture (don’t miss the original Starbucks!).

15. **Attend a Sports Game**: Catch a Seahawks (NFL), Mariners (MLB), or Sounders (MLS) game for a lively local experience.

Whether you're interested in culture, nature, food, or history, Seattle has something for everyone to enjoy!
```

The above `SimpleAgent` always responds with a fresh context that contains only
the system message and the latest user's message.
We can use model context classes from {py:mod}`autogen_core.model_context`
to make the agent "remember" previous conversations.
See the [Model Context](./model-context.ipynb) page for more details.

#### API Keys From Environment Variables

In the examples above, we show that you can provide the API key through the `api_key` argument. Importantly, the OpenAI and Azure OpenAI clients use the [openai package](https://github.com/openai/openai-python/blob/3f8d8205ae41c389541e125336b0ae0c5e437661/src/openai/__init__.py#L260), which will automatically read an api key from the environment variable if one is not provided.

- For OpenAI, you can set the `OPENAI_API_KEY` environment variable.  
- For Azure OpenAI, you can set the `AZURE_OPENAI_API_KEY` environment variable. 

In addition, for Gemini (Beta), you can set the `GEMINI_API_KEY` environment variable.

This is a good practice to explore, as it avoids including sensitive api keys in your code.


## Model Context

*[components/model-context.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-context.html)*

#### Model Context

A model context supports storage and retrieval of Chat Completion messages.
It is always used together with a model client to generate LLM-based responses.

For example, {py:mod}`~autogen_core.model_context.BufferedChatCompletionContext`
is a most-recent-used (MRU) context that stores the most recent `buffer_size`
number of messages. This is useful to avoid context overflow in many LLMs.

Let's see an example that uses
{py:mod}`~autogen_core.model_context.BufferedChatCompletionContext`.

```python
from dataclasses import dataclass

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

```python
@dataclass
class Message:
    content: str
```

```python
class SimpleAgentWithContext(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Prepare input to the chat completion model.
        user_message = UserMessage(content=message.content, source="user")
        # Add message to model context.
        await self._model_context.add_message(user_message)
        # Generate a response.
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        # Return with the model's response.
        assert isinstance(response.content, str)
        # Add message to model context.
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))
        return Message(content=response.content)
```

Now let's try to ask follow up questions after the first one.

```python
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY set in the environment.
)

runtime = SingleThreadedAgentRuntime()
await SimpleAgentWithContext.register(
    runtime,
    "simple_agent_context",
    lambda: SimpleAgentWithContext(model_client=model_client),
)
# Start the runtime processing messages.
runtime.start()
agent_id = AgentId("simple_agent_context", "default")

# First question.
message = Message("Hello, what are some fun things to do in Seattle?")
print(f"Question: {message.content}")
response = await runtime.send_message(message, agent_id)
print(f"Response: {response.content}")
print("-----")

# Second question.
message = Message("What was the first thing you mentioned?")
print(f"Question: {message.content}")
response = await runtime.send_message(message, agent_id)
print(f"Response: {response.content}")

# Stop the runtime processing messages.
await runtime.stop()
await model_client.close()
```

```text
Question: Hello, what are some fun things to do in Seattle?
Response: Seattle offers a variety of fun activities and attractions. Here are some highlights:

1. **Pike Place Market**: Visit this iconic market to explore local vendors, fresh produce, artisanal products, and watch the famous fish throwing.

2. **Space Needle**: Take a trip to the observation deck for stunning panoramic views of the city, Puget Sound, and the surrounding mountains.

3. **Chihuly Garden and Glass**: Marvel at the stunning glass art installations created by artist Dale Chihuly, located right next to the Space Needle.

4. **Seattle Waterfront**: Enjoy a stroll along the waterfront, visit the Seattle Aquarium, and take a ferry ride to nearby islands like Bainbridge Island.

5. **Museum of Pop Culture (MoPOP)**: Explore exhibits on music, science fiction, and pop culture in this architecturally striking building.

6. **Seattle Art Museum (SAM)**: Discover an extensive collection of art from around the world, including contemporary and Native American art.

7. **Gas Works Park**: Relax in this unique park that features remnants of an old gasification plant, offering great views of the Seattle skyline and Lake Union.

8. **Discovery Park**: Enjoy nature trails, beaches, and beautiful views of the Puget Sound and the Olympic Mountains in this large urban park.

9. **Ballard Locks**: Watch boats navigate the locks and see fish swimming upstream during the salmon migration season.

10. **Fremont Troll**: Check out this quirky public art installation under a bridge in the Fremont neighborhood.

11. **Underground Tour**: Take an entertaining guided tour through the underground passages of Pioneer Square to learn about Seattle's history.

12. **Brewery Tours**: Seattle is known for its craft beer scene. Visit local breweries for tastings and tours.

13. **Seattle Center**: Explore the cultural complex that includes the Space Needle, MoPOP, and various festivals and events throughout the year.

These are just a few options, and Seattle has something for everyone, whether you're into outdoor activities, culture, history, or food!
-----
Question: What was the first thing you mentioned?
Response: The first thing I mentioned was **Pike Place Market**. It's an iconic market in Seattle known for its local vendors, fresh produce, artisanal products, and the famous fish throwing by the fishmongers. It's a vibrant place full of sights, sounds, and delicious food.
```

From the second response, you can see the agent now can recall its own previous responses.


## Tools

*[components/tools.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/tools.html)*

### Tools

Tools are code that can be executed by an agent to perform actions. A tool
can be a simple function such as a calculator, or an API call to a third-party service
such as stock price lookup or weather forecast.
In the context of AI agents, tools are designed to be executed by agents in
response to model-generated function calls.

AutoGen provides the {py:mod}`autogen_core.tools` module with a suite of built-in
tools and utilities for creating and running custom tools.

#### Built-in Tools

One of the built-in tools is the {py:class}`~autogen_ext.tools.code_execution.PythonCodeExecutionTool`,
which allows agents to execute Python code snippets.

Here is how you create the tool and use it.

```python
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

# Create the tool.
code_executor = DockerCommandLineCodeExecutor()
await code_executor.start()
code_execution_tool = PythonCodeExecutionTool(code_executor)
cancellation_token = CancellationToken()

# Use the tool directly without an agent.
code = "print('Hello, world!')"
result = await code_execution_tool.run_json({"code": code}, cancellation_token)
print(code_execution_tool.return_value_as_string(result))
```

```text
Hello, world!
```

The {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
class is a built-in code executor that runs Python code snippets in a subprocess
in the command line environment of a docker container.
The {py:class}`~autogen_ext.tools.code_execution.PythonCodeExecutionTool` class wraps the code executor
and provides a simple interface to execute Python code snippets.

Examples of other built-in tools
- {py:class}`~autogen_ext.tools.graphrag.LocalSearchTool` and {py:class}`~autogen_ext.tools.graphrag.GlobalSearchTool` for using [GraphRAG](https://github.com/microsoft/graphrag).
- {py:class}`~autogen_ext.tools.mcp.mcp_server_tools` for using [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) servers as tools.
- {py:class}`~autogen_ext.tools.http.HttpTool` for making HTTP requests to REST APIs.
- {py:class}`~autogen_ext.tools.langchain.LangChainToolAdapter` for using LangChain tools.

#### Custom Function Tools

A tool can also be a simple Python function that performs a specific action.
To create a custom function tool, you just need to create a Python function
and use the {py:class}`~autogen_core.tools.FunctionTool` class to wrap it.

The {py:class}`~autogen_core.tools.FunctionTool` class uses descriptions and type annotations
to inform the LLM when and how to use a given function. The description provides context
about the function’s purpose and intended use cases, while type annotations inform the LLM about
the expected parameters and return type.

For example, a simple tool to obtain the stock price of a company might look like this:

```python
import random

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from typing_extensions import Annotated


async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    # Returns a random stock price for demonstration purposes.
    return random.uniform(10, 200)


# Create a function tool.
stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

# Run the tool.
cancellation_token = CancellationToken()
result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token)

# Print the result.
print(stock_price_tool.return_value_as_string(result))
```

```text
143.83831971965762
```

#### Calling Tools with Model Clients

In AutoGen, every tool is a subclass of {py:class}`~autogen_core.tools.BaseTool`,
which automatically generates the JSON schema for the tool.
For example, to get the JSON schema for the `stock_price_tool`, we can use the
{py:attr}`~autogen_core.tools.BaseTool.schema` property.

```python
stock_price_tool.schema
```

```text
{'name': 'get_stock_price',
 'description': 'Get the stock price.',
 'parameters': {'type': 'object',
  'properties': {'ticker': {'description': 'ticker',
    'title': 'Ticker',
    'type': 'string'},
   'date': {'description': 'Date in YYYY/MM/DD',
    'title': 'Date',
    'type': 'string'}},
  'required': ['ticker', 'date'],
  'additionalProperties': False},
 'strict': False}
```

Model clients use the JSON schema of the tools to generate tool calls.

Here is an example of how to use the {py:class}`~autogen_core.tools.FunctionTool` class
with a {py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient`.
Other model client classes can be used in a similar way. See [Model Clients](./model-clients.ipynb)
for more details.

```python
import json

from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create the OpenAI chat completion client. Using OPENAI_API_KEY from environment variable.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create a user message.
user_message = UserMessage(content="What is the stock price of AAPL on 2021/01/01?", source="user")

# Run the chat completion with the stock_price_tool defined above.
cancellation_token = CancellationToken()
create_result = await model_client.create(
    messages=[user_message], tools=[stock_price_tool], cancellation_token=cancellation_token
)
create_result.content
```

```text
[FunctionCall(id='call_tpJ5J1Xoxi84Sw4v0scH0qBM', arguments='{"ticker":"AAPL","date":"2021/01/01"}', name='get_stock_price')]
```

What is actually going on under the hood of the call to the
{py:class}`~autogen_ext.models.openai.BaseOpenAIChatCompletionClient.create`
method? The model client takes the list of tools and generates a JSON schema
for the parameters of each tool. Then, it generates a request to the model
API with the tool's JSON schema and the other messages to obtain a result.

Many models, such as OpenAI's GPT-4o and Llama-3.2, are trained to produce
tool calls in the form of structured JSON strings that conform to the
JSON schema of the tool. AutoGen's model clients then parse the model's response
and extract the tool call from the JSON string.

The result is a list of {py:class}`~autogen_core.FunctionCall` objects, which can be
used to run the corresponding tools.

We use `json.loads` to parse the JSON string in the {py:class}`~autogen_core.FunctionCall.arguments`
field into a Python dictionary. The {py:meth}`~autogen_core.tools.BaseTool.run_json`
method takes the dictionary and runs the tool with the provided arguments.

```python
assert isinstance(create_result.content, list)
arguments = json.loads(create_result.content[0].arguments)  # type: ignore
tool_result = await stock_price_tool.run_json(arguments, cancellation_token)
tool_result_str = stock_price_tool.return_value_as_string(tool_result)
tool_result_str
```

```text
'32.381250753393104'
```

Now you can make another model client call to have the model generate a reflection
on the result of the tool execution.

The result of the tool call is wrapped in a {py:class}`~autogen_core.models.FunctionExecutionResult`
object, which contains the result of the tool execution and the ID of the tool that was called.
The model client can use this information to generate a reflection on the result of the tool execution.

```python
# Create a function execution result
exec_result = FunctionExecutionResult(
    call_id=create_result.content[0].id,  # type: ignore
    content=tool_result_str,
    is_error=False,
    name=stock_price_tool.name,
)

# Make another chat completion with the history and function execution result message.
messages = [
    user_message,
    AssistantMessage(content=create_result.content, source="assistant"),  # assistant message with tool call
    FunctionExecutionResultMessage(content=[exec_result]),  # function execution result message
]
create_result = await model_client.create(messages=messages, cancellation_token=cancellation_token)  # type: ignore
print(create_result.content)
await model_client.close()
```

```text
The stock price of AAPL (Apple Inc.) on January 1, 2021, was approximately $32.38.
```

#### Tool-Equipped Agent

Putting the model client and the tools together, you can create a tool-equipped agent
that can use tools to perform actions, and reflect on the results of those actions.

```{note}
The Core API is designed to be minimal and you need to build your own agent logic around model clients and tools.
For "pre-built" agents that can use tools, please refer to the [AgentChat API](../../agentchat-user-guide/index.md).
```

```python
import asyncio
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    AgentId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)
```

When handling a user message, the `ToolUseAgent` class first use the model client
to generate a list of function calls to the tools, and then run the tools
and generate a reflection on the results of the tool execution.
The reflection is then returned to the user as the agent's response.

To run the agent, let's create a runtime and register the agent with the runtime.

```python
# Create the model client.
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
# Create a runtime.
runtime = SingleThreadedAgentRuntime()
# Create the tools.
tools: List[Tool] = [FunctionTool(get_stock_price, description="Get the stock price.")]
# Register the agents.
await ToolUseAgent.register(
    runtime,
    "tool_use_agent",
    lambda: ToolUseAgent(
        model_client=model_client,
        tool_schema=tools,
    ),
)
```

```text
AgentType(type='tool_use_agent')
```

This example uses the {py:class}`~autogen_ext.models.openai.OpenAIChatCompletionClient`,
for Azure OpenAI and other clients, see [Model Clients](./model-clients.ipynb).
Let's test the agent with a question about stock price.

```python
# Start processing messages.
runtime.start()
# Send a direct message to the tool agent.
tool_use_agent = AgentId("tool_use_agent", "default")
response = await runtime.send_message(Message("What is the stock price of NVDA on 2024/06/01?"), tool_use_agent)
print(response.content)
# Stop processing messages.
await runtime.stop()
await model_client.close()
```

```text
The stock price of NVIDIA (NVDA) on June 1, 2024, was approximately $140.05.
```


## Workbench (and MCP)

*[components/workbench.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/workbench.html)*

### Workbench (and MCP)

A {py:class}`~autogen_core.tools.Workbench` provides a collection of tools that share state and resources.
Different from {py:class}`~autogen_core.tools.Tool`, which provides an interface
to a single tool, a workbench provides an interface to call different tools
and receive results as the same types.

#### Using Workbench

Here is an example of how to create an agent using {py:class}`~autogen_core.tools.Workbench`.

```python
import json
from dataclasses import dataclass
from typing import List

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import ToolResult, Workbench
```

```python
@dataclass
class Message:
    content: str


class WorkbenchAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, model_context: ChatCompletionContext, workbench: Workbench
    ) -> None:
        super().__init__("An agent with a workbench")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Add the user message to the model context.
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        print("---------User Message-----------")
        print(message.content)

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=(await self._workbench.list_tools()),
            cancellation_token=ctx.cancellation_token,
        )

        # Run tool call loop.
        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            print("---------Function Calls-----------")
            for call in create_result.content:
                print(call)

            # Add the function calls to the model context.
            await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

            # Call the tools using the workbench.
            print("---------Function Call Results-----------")
            results: List[ToolResult] = []
            for call in create_result.content:
                result = await self._workbench.call_tool(
                    call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                )
                results.append(result)
                print(result)

            # Add the function execution results to the model context.
            await self._model_context.add_message(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
            )

            # Run the chat completion again to reflect on the history and function execution results.
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )

        # Now we have a single message as the result.
        assert isinstance(create_result.content, str)

        print("---------Final Response-----------")
        print(create_result.content)

        # Add the assistant message to the model context.
        await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))

        # Return the result as a message.
        return Message(content=create_result.content)
```

In this example, the agent calls the tools provided by the workbench
in a loop until the model returns a final answer.

#### MCP Workbench

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a protocol
for providing tools and resources
to language models. An MCP server hosts a set of tools and manages their state,
while an MCP client operates from the side of the language model and
communicates with the server to access the tools, and to provide the
language model with the context it needs to use the tools effectively.

In AutoGen, we provide {py:class}`~autogen_ext.tools.mcp.McpWorkbench`
that implements an MCP client. You can use it to create an agent that
uses tools provided by MCP servers.

#### Web Browsing Agent using Playwright MCP

Here is an example of how we can use the [Playwright MCP server](https://github.com/microsoft/playwright-mcp)
and the `WorkbenchAgent` class to create a web browsing agent.

You may need to install the browser dependencies for Playwright.

```python
# npx playwright install chrome
```

Start the Playwright MCP server in a terminal.

```python
# npx @playwright/mcp@latest --port 8931
```

Then, create the agent using the `WorkbenchAgent` class and
{py:class}`~autogen_ext.tools.mcp.McpWorkbench` with the Playwright MCP server URL.

```python
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams

playwright_server_params = SseServerParams(
    url="http://localhost:8931/sse",
)

# Start the workbench in a context manager.
# You can also start and stop the workbench using `workbench.start()` and `workbench.stop()`.
async with McpWorkbench(playwright_server_params) as workbench:  # type: ignore
    # Create a single-threaded agent runtime.
    runtime = SingleThreadedAgentRuntime()

    # Register the agent with the runtime.
    await WorkbenchAgent.register(
        runtime=runtime,
        type="WebAgent",
        factory=lambda: WorkbenchAgent(
            model_client=OpenAIChatCompletionClient(model="gpt-4.1-nano"),
            model_context=BufferedChatCompletionContext(buffer_size=10),
            workbench=workbench,
        ),
    )

    # Start the runtime.
    runtime.start()

    # Send a message to the agent.
    await runtime.send_message(
        Message(content="Use Bing to find out the address of Microsoft Building 99"),
        recipient=AgentId("WebAgent", "default"),
    )

    # Stop the runtime.
    await runtime.stop()
```

````text
---------User Message-----------
Use Bing to find out the address of Microsoft Building 99
---------Function Calls-----------
FunctionCall(id='call_oJl0E0hWvmKZrzAM7huiIyus', arguments='{"url": "https://www.bing.com"}', name='browser_navigate')
FunctionCall(id='call_Qfab5bAsveZIVg2v0aHl4Kgv', arguments='{}', name='browser_snapshot')
---------Function Call Results-----------
type='ToolResult' name='browser_navigate' result=[TextResultContent(type='TextResultContent', content='- Ran Playwright code:\n```js\n// Navigate to https://www.bing.com\nawait page.goto(\'https://www.bing.com\');\n```\n\n- Page URL: https://www.bing.com/\n- Page Title: Search - Microsoft Bing\n- Page Snapshot\n```yaml\n- generic [ref=s1e2]:\n  - generic [ref=s1e4]:\n    - generic:\n      - generic [ref=s1e6]:\n        - generic [ref=s1e7]\n        - generic [ref=s1e10]:\n          - img "Background image" [ref=s1e12]\n      - generic [ref=s1e14]:\n        - generic [ref=s1e17]\n        - generic [ref=s1e18]:\n          - img "Background image" [ref=s1e20]\n    - main [ref=s1e23]:\n      - generic [ref=s1e24]:\n        - generic [ref=s1e25]:\n          - heading "Trending Now on Bing" [level=1] [ref=s1e26]\n          - navigation [ref=s1e27]:\n            - menubar [ref=s1e28]:\n              - menuitem "Copilot" [ref=s1e29]:\n                - link "Copilot" [ref=s1e30]:\n                  - /url: /chat?FORM=hpcodx\n                  - text: Copilot\n              - menuitem "Images" [ref=s1e34]:\n                - link "Images" [ref=s1e35]:\n                  - /url: /images?FORM=Z9LH\n              - menuitem "Videos" [ref=s1e36]:\n                - link "Videos" [ref=s1e37]:\n                  - /url: /videos?FORM=Z9LH1\n              - menuitem "Shopping" [ref=s1e38]:\n                - link "Shopping" [ref=s1e39]:\n                  - /url: /shop?FORM=Z9LHS4\n              - menuitem "Maps" [ref=s1e40]:\n                - link "Maps" [ref=s1e41]:\n                  - /url: /maps?FORM=Z9LH2\n              - menuitem "News" [ref=s1e42]:\n                - link "News" [ref=s1e43]:\n                  - /url: /news/search?q=Top+stories&nvaug=%5bNewsVertical+Category%3d%22rt_MaxClass%22%5d&FORM=Z9LH3\n              - menuitem ". . . More" [ref=s1e44]:\n                - text: . . .\n                - tooltip "More" [ref=s1e45]\n        - generic\n      - generic [ref=s1e49]:\n        - search [ref=s1e50]:\n          - generic [ref=s1e52]:\n            - textbox "0 characters out of 2000" [ref=s1e53]\n          - button "Search using voice" [ref=s1e55]:\n            - img [ref=s1e56]\n            - text: Search using voice\n        - link "Open Copilot" [ref=s1e61]:\n          - /url: /chat?FORM=hpcodx\n          - generic [ref=s1e63]\n    - generic\n    - generic [ref=s1e67]:\n      - generic [ref=s1e69]:\n        - generic [ref=s1e71]:\n          - generic:\n            - link "Get the new Bing Wallpaper app":\n              - /url: https://go.microsoft.com/fwlink/?linkid=2127455\n              - text: Get the new Bing Wallpaper app\n            - \'heading "Image of the day: Spire Cove in Kenai Fjords National Park, Seward, Alaska" [level=3]\':\n              - \'link "Image of the day: Spire Cove in Kenai Fjords National Park, Seward, Alaska"\':\n                - /url: /search?q=Kenai+Fjords+National+Park+Alaska&form=hpcapt&filters=HpDate:"20250424_0700"\n                - text: Spire Cove in Kenai Fjords National Park, Seward, Alaska\n            - generic:\n              - text: © Wander Photography/Getty Images\n              - list:\n                - listitem:\n                  - button "Download this image. Use of this image is restricted\n                    to wallpaper only."\n          - generic [ref=s1e84]:\n            - link "Rugged peaks and wild waters" [ref=s1e86]:\n              - /url: /search?q=Kenai+Fjords+National+Park+Alaska&form=hpcapt&filters=HpDate:"20250424_0700"\n              - heading "Rugged peaks and wild waters" [level=2] [ref=s1e88]\n            - generic [ref=s1e89]:\n              - button "Previous image" [disabled] [ref=s1e90]\n              - button "Next image" [disabled] [ref=s1e91]\n        - button "Feedback" [ref=s1e92]:\n          - img [ref=s1e93]\n          - text: Feedback\n        - complementary\n```')] is_error=False
type='ToolResult' name='browser_snapshot' result=[TextResultContent(type='TextResultContent', content='- Ran Playwright code:\n```js\n// <internal code to capture accessibility snapshot>\n```\n\n- Page URL: https://www.bing.com/\n- Page Title: Search - Microsoft Bing\n- Page Snapshot\n```yaml\n- generic [ref=s2e2]:\n  - generic [ref=s2e4]:\n    - generic:\n      - generic [ref=s2e6]:\n        - generic [ref=s2e7]\n        - generic [ref=s2e10]:\n          - img "Background image" [ref=s2e12]\n      - generic [ref=s2e14]:\n        - generic [ref=s2e17]\n        - generic [ref=s2e18]:\n          - img "Background image" [ref=s2e20]\n    - main [ref=s2e23]:\n      - generic [ref=s2e24]:\n        - generic [ref=s2e25]:\n          - heading "Trending Now on Bing" [level=1] [ref=s2e26]\n          - navigation [ref=s2e27]:\n            - menubar [ref=s2e28]:\n              - menuitem "Copilot" [ref=s2e29]:\n                - link "Copilot" [ref=s2e30]:\n                  - /url: /chat?FORM=hpcodx\n                  - text: Copilot\n              - menuitem "Images" [ref=s2e34]:\n                - link "Images" [ref=s2e35]:\n                  - /url: /images?FORM=Z9LH\n              - menuitem "Videos" [ref=s2e36]:\n                - link "Videos" [ref=s2e37]:\n                  - /url: /videos?FORM=Z9LH1\n              - menuitem "Shopping" [ref=s2e38]:\n                - link "Shopping" [ref=s2e39]:\n                  - /url: /shop?FORM=Z9LHS4\n              - menuitem "Maps" [ref=s2e40]:\n                - link "Maps" [ref=s2e41]:\n                  - /url: /maps?FORM=Z9LH2\n              - menuitem "News" [ref=s2e42]:\n                - link "News" [ref=s
````


## Command Line Code Executors

*[components/command-line-code-executors.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/command-line-code-executors.html)*

### Command Line Code Executors

Command line code execution is the simplest form of code execution.
Generally speaking, it will save each code block to a file and then execute that file.
This means that each code block is executed in a new process. There are two forms of this executor:

- Docker ({py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`) - this is where all commands are executed in a Docker container
- Local ({py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`) - this is where all commands are executed on the host machine

#### Docker

```{note}
To use {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`, ensure the `autogen-ext[docker]` package is installed. For more details, see the [Packages Documentation](https://microsoft.github.io/autogen/dev/packages/index.html).

```

The {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` will create a Docker container and run all commands within that container. 
The default image that is used is `python:3-slim`, this can be customized by passing the `image` parameter to the constructor. 
If the image is not found locally then the class will try to pull it. 
Therefore, having built the image locally is enough. The only thing required for this image to be compatible with the executor is to have `sh` and `python` installed. 
Therefore, creating a custom image is a simple and effective way to ensure required system dependencies are available.

You can use the executor as a context manager to ensure the container is cleaned up after use. 
Otherwise, the `atexit` module will be used to stop the container when the program exits.

##### Inspecting the container

If you wish to keep the container around after AutoGen is finished using it for whatever reason (e.g. to inspect the container), 
then you can set the `auto_remove` parameter to `False` when creating the executor. 
`stop_container` can also be set to `False` to prevent the container from being stopped at the end of the execution.

##### Example

```python
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:  # type: ignore
    print(
        await executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code="print('Hello, World!')"),
            ],
            cancellation_token=CancellationToken(),
        )
    )
```

```text
CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='coding/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.python')
```

##### Combining an Application in Docker with a Docker based executor

It is desirable to bundle your application into a Docker image. But then, how do you allow your containerised application to execute code in a different container?

The recommended approach to this is called "Docker out of Docker", where the Docker socket is mounted to the main AutoGen container, so that it can spawn and control "sibling" containers on the host. This is better than what is called "Docker in Docker", where the main container runs a Docker daemon and spawns containers within itself. You can read more about this [here](https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/).

To do this you would need to mount the Docker socket into the container running your application. This can be done by adding the following to the `docker run` command:

```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

This will allow your application's container to spawn and control sibling containers on the host.

If you need to bind a working directory to the application's container but the directory belongs to your host machine, 
use the `bind_dir` parameter. This will allow the application's container to bind the *host* directory to the new spawned containers and allow it to access the files within the said directory. If the `bind_dir` is not specified, it will fallback to `work_dir`.

#### Local

```{attention}
The local version will run code on your local system. Use it with caution.
```

To execute code on the host machine, as in the machine running your application, {py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor` can be used.

##### Example

```python
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
print(
    await local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
)
```

```text
CommandLineCodeResult(exit_code=0, output='Hello, World!\n', code_file='/home/ekzhu/agnext/python/packages/autogen-core/docs/src/guides/coding/tmp_code_07da107bb575cc4e02b0e1d6d99cc204.py')
```

#### Local within a Virtual Environment

If you want the code to run within a virtual environment created as part of the application’s setup, you can specify a directory for the newly created environment and pass its context to  {py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`. This setup allows the executor to use the specified virtual environment consistently throughout the application's lifetime, ensuring isolated dependencies and a controlled runtime environment.

```python
import venv
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

venv_dir = work_dir / ".venv"
venv_builder = venv.EnvBuilder(with_pip=True)
venv_builder.create(venv_dir)
venv_context = venv_builder.ensure_directories(venv_dir)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir, virtual_env_context=venv_context)
await local_executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="bash", code="pip install matplotlib"),
    ],
    cancellation_token=CancellationToken(),
)
```

```text
CommandLineCodeResult(exit_code=0, output='', code_file='/Users/gziz/Dev/autogen/python/packages/autogen-core/docs/src/user-guide/core-user-guide/framework/coding/tmp_code_d2a7db48799db3cc785156a11a38822a45c19f3956f02ec69b92e4169ecbf2ca.bash')
```

As we can see, the code has executed successfully, and the installation has been isolated to the newly created virtual environment, without affecting our global environment.


---

# Multi-Agent Design Patterns


## Intro

*[design-patterns/intro.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/intro.html)*

### Intro

Agents can work together in a variety of ways to solve problems.
Research works like [AutoGen](https://aka.ms/autogen-paper),
[MetaGPT](https://arxiv.org/abs/2308.00352)
and [ChatDev](https://arxiv.org/abs/2307.07924) have shown
multi-agent systems out-performing single agent systems at complex tasks
like software development.

A multi-agent design pattern is a structure that emerges from message protocols:
it describes how agents interact with each other to solve problems.
For example, the [tool-equipped agent](../components/tools.ipynb#tool-equipped-agent) in
the previous section employs a design pattern called ReAct,
which involves an agent interacting with tools.

You can implement any multi-agent design pattern using AutoGen agents.
In the next two sections, we will discuss two common design patterns:
group chat for task decomposition, and reflection for robustness.





## Concurrent Agents

*[design-patterns/concurrent-agents.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/concurrent-agents.html)*

### Concurrent Agents

In this section, we explore the use of multiple agents working concurrently. We cover three main patterns:

1. **Single Message & Multiple Processors**  
   Demonstrates how a single message can be processed by multiple agents subscribed to the same topic simultaneously.

2. **Multiple Messages & Multiple Processors**  
   Illustrates how specific message types can be routed to dedicated agents based on topics.

3. **Direct Messaging**  
   Focuses on sending messages between agents and from the runtime to agents.

```python
import asyncio
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
    type_subscription,
)
```

```python
@dataclass
class Task:
    task_id: str


@dataclass
class TaskResponse:
    task_id: str
    result: str
```

#### Single Message & Multiple Processors
The first pattern shows how a single message can be processed by multiple agents simultaneously:

- Each `Processor` agent subscribes to the default topic using the {py:meth}`~autogen_core.components.default_subscription` decorator.
- When publishing a message to the default topic, all registered agents will process the message independently.
```{note}
Below, we are subscribing `Processor` using the {py:meth}`~autogen_core.components.default_subscription` decorator, there's an alternative way to subscribe an agent without using decorators altogether as shown in [Subscribe and Publish to Topics](../framework/message-and-communication.ipynb#subscribe-and-publish-to-topics), this way the same agent class can be subscribed to different topics.
```

```python
@default_subscription
class Processor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"{self._description} starting task {message.task_id}")
        await asyncio.sleep(2)  # Simulate work
        print(f"{self._description} finished task {message.task_id}")
```

```python
runtime = SingleThreadedAgentRuntime()

await Processor.register(runtime, "agent_1", lambda: Processor("Agent 1"))
await Processor.register(runtime, "agent_2", lambda: Processor("Agent 2"))

runtime.start()

await runtime.publish_message(Task(task_id="task-1"), topic_id=DefaultTopicId())

await runtime.stop_when_idle()
```

```text
Agent 1 starting task task-1
Agent 2 starting task task-1
Agent 1 finished task task-1
Agent 2 finished task task-1
```

#### Multiple messages & Multiple Processors
Second, this pattern demonstrates routing different types of messages to specific processors:
- `UrgentProcessor` subscribes to the "urgent" topic
- `NormalProcessor` subscribes to the "normal" topic

We make an agent subscribe to a specific topic type using the {py:meth}`~autogen_core.components.type_subscription` decorator.

```python
TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")


@type_subscription(topic_type="urgent")
class UrgentProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Urgent processor starting task {message.task_id}")
        await asyncio.sleep(1)  # Simulate work
        print(f"Urgent processor finished task {message.task_id}")

        task_response = TaskResponse(task_id=message.task_id, result="Results by Urgent Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)


@type_subscription(topic_type="normal")
class NormalProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Normal processor starting task {message.task_id}")
        await asyncio.sleep(3)  # Simulate work
        print(f"Normal processor finished task {message.task_id}")

        task_response = TaskResponse(task_id=message.task_id, result="Results by Normal Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)
```

After registering the agents, we can publish messages to the "urgent" and "normal" topics:

```python
runtime = SingleThreadedAgentRuntime()

await UrgentProcessor.register(runtime, "urgent_processor", lambda: UrgentProcessor("Urgent Processor"))
await NormalProcessor.register(runtime, "normal_processor", lambda: NormalProcessor("Normal Processor"))

runtime.start()

await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

await runtime.stop_when_idle()
```

```text
Normal processor starting task normal-1
Urgent processor starting task urgent-1
Urgent processor finished task urgent-1
Normal processor finished task normal-1
```

###### Collecting Results

In the previous example, we relied on console printing to verify task completion. However, in real applications, we typically want to collect and process the results programmatically.

To collect these messages, we'll use a {py:class}`~autogen_core.components.ClosureAgent`. We've defined a dedicated topic `TASK_RESULTS_TOPIC_TYPE` where both `UrgentProcessor` and `NormalProcessor` publish their results. The ClosureAgent will then process messages from this topic.

```python
queue = asyncio.Queue[TaskResponse]()


async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
    await queue.put(message)


runtime.start()

CLOSURE_AGENT_TYPE = "collect_result_agent"
await ClosureAgent.register_closure(
    runtime,
    CLOSURE_AGENT_TYPE,
    collect_result,
    subscriptions=lambda: [TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)],
)

await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

await runtime.stop_when_idle()
```

```text
Normal processor starting task normal-1
Urgent processor starting task urgent-1
Urgent processor finished task urgent-1
Normal processor finished task normal-1
```

```python
while not queue.empty():
    print(await queue.get())
```

```text
TaskResponse(task_id='urgent-1', result='Results by Urgent Processor')
TaskResponse(task_id='normal-1', result='Results by Normal Processor')
```

#### Direct Messages

In contrast to the previous patterns, this pattern focuses on direct messages. Here we demonstrate two ways to send them:

- Direct messaging between agents  
- Sending messages from the runtime to specific agents  

Things to consider in the example below:

- Messages are addressed using the {py:class}`~autogen_core.components.AgentId`.  
- The sender can expect to receive a response from the target agent.  
- We register the `WorkerAgent` class only once; however, we send tasks to two different workers.
    - How? As stated in [Agent lifecycle](../core-concepts/agent-identity-and-lifecycle.md#agent-lifecycle), when delivering a message using an {py:class}`~autogen_core.components.AgentId`, the runtime will either fetch the instance or create one if it doesn't exist. In this case, the runtime creates two instances of workers when sending those two messages.

```python
class WorkerAgent(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> TaskResponse:
        print(f"{self.id} starting task {message.task_id}")
        await asyncio.sleep(2)  # Simulate work
        print(f"{self.id} finished task {message.task_id}")
        return TaskResponse(task_id=message.task_id, result=f"Results by {self.id}")


class DelegatorAgent(RoutedAgent):
    def __init__(self, description: str, worker_type: str):
        super().__init__(description)
        self.worker_instances = [AgentId(worker_type, f"{worker_type}-1"), AgentId(worker_type, f"{worker_type}-2")]

    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> TaskResponse:
        print(f"Delegator received task {message.task_id}.")

        subtask1 = Task(task_id="task-part-1")
        subtask2 = Task(task_id="task-part-2")

        worker1_result, worker2_result = await asyncio.gather(
            self.send_message(subtask1, self.worker_instances[0]), self.send_message(subtask2, self.worker_instances[1])
        )

        combined_result = f"Part 1: {worker1_result.result}, " f"Part 2: {worker2_result.result}"
        task_response = TaskResponse(task_id=message.task_id, result=combined_result)
        return task_response
```

```python
runtime = SingleThreadedAgentRuntime()

await WorkerAgent.register(runtime, "worker", lambda: WorkerAgent("Worker Agent"))
await DelegatorAgent.register(runtime, "delegator", lambda: DelegatorAgent("Delegator Agent", "worker"))

runtime.start()

delegator = AgentId("delegator", "default")
response = await runtime.send_message(Task(task_id="main-task"), recipient=delegator)

print(f"Final result: {response.result}")
await runtime.stop_when_idle()
```

```text
Delegator received task main-task.
worker/worker-1 starting task task-part-1
worker/worker-2 starting task task-part-2
worker/worker-1 finished task task-part-1
worker/worker-2 finished task task-part-2
Final result: Part 1: Results by worker/worker-1, Part 2: Results by worker/worker-2
```

#### Additional Resources

If you're interested in more about concurrent processing, check out the [Mixture of Agents](./mixture-of-agents.ipynb) pattern, which relies heavily on concurrent agents.


## Sequential Workflow

*[design-patterns/sequential-workflow.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/sequential-workflow.html)*

### Sequential Workflow

Sequential Workflow is a multi-agent design pattern where agents respond in a deterministic sequence. Each agent in the workflow performs a specific task by processing a message, generating a response, and then passing it to the next agent. This pattern is useful for creating deterministic workflows where each agent contributes to a pre-specified sub-task.

In this example, we demonstrate a sequential workflow where multiple agents collaborate to transform a basic product description into a polished marketing copy.

The pipeline consists of four specialized agents:
- **Concept Extractor Agent**: Analyzes the initial product description to extract key features, target audience, and unique selling points (USPs). The output is a structured analysis in a single text block.
- **Writer Agent**: Crafts compelling marketing copy based on the extracted concepts. This agent transforms the analytical insights into engaging promotional content, delivering a cohesive narrative in a single text block.
- **Format & Proof Agent**: Polishes the draft copy by refining grammar, enhancing clarity, and maintaining consistent tone. This agent ensures professional quality and delivers a well-formatted final version.
- **User Agent**: Presents the final, refined marketing copy to the user, completing the workflow.

The following diagram illustrates the sequential workflow in this example:

![Sequential Workflow](sequential-workflow.svg)

We will implement this workflow using publish-subscribe messaging.
Please read about [Topic and Subscription](../core-concepts/topic-and-subscription.md) for the core concepts
and [Broadcast Messaging](../framework/message-and-communication.ipynb#broadcast) for the the API usage.

In this pipeline, agents communicate with each other by publishing their completed work as messages to the topic of the 
next agent in the sequence. For example, when the `ConceptExtractor` finishes analyzing the product description, it 
publishes its findings to the `"WriterAgent"` topic, which the `WriterAgent` is subscribed to. This pattern continues through 
each step of the pipeline, with each agent publishing to the topic that the next agent in line subscribed to.

```python
from dataclasses import dataclass

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

#### Message Protocol

The message protocol for this example workflow is a simple text message that agents will use to relay their work.

```python
@dataclass
class Message:
    content: str
```

#### Topics

Each agent in the workflow will be subscribed to a specific topic type. The topic types are named after the agents in the sequence,
This allows each agent to publish its work to the next agent in the sequence.

```python
concept_extractor_topic_type = "ConceptExtractorAgent"
writer_topic_type = "WriterAgent"
format_proof_topic_type = "FormatProofAgent"
user_topic_type = "User"
```

#### Agents

Each agent class is defined with a {py:class}`~autogen_core.type_subscription` decorator to specify the topic type it is subscribed to.
Alternative to the decorator, you can also use the {py:meth}`~autogen_core.AgentRuntime.add_subscription` method to subscribe to a topic through runtime directly.

The concept extractor agent comes up with the initial bullet points for the product description.

```python
@type_subscription(topic_type=concept_extractor_topic_type)
class ConceptExtractorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A concept extractor agent.")
        self._system_message = SystemMessage(
            content=(
                "You are a marketing analyst. Given a product description, identify:\n"
                "- Key features\n"
                "- Target audience\n"
                "- Unique selling points\n\n"
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_user_description(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Product description: {message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(writer_topic_type, source=self.id.key))
```

The writer agent performs writing.

```python
@type_subscription(topic_type=writer_topic_type)
class WriterAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A writer agent.")
        self._system_message = SystemMessage(
            content=(
                "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
                "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
                "Output should be short (around 150 words), output just the copy as a single text block."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_intermediate_text(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Below is the info about the product:\n\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(format_proof_topic_type, source=self.id.key))
```

The format proof agent performs the formatting.

```python
@type_subscription(topic_type=format_proof_topic_type)
class FormatProofAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A format & proof agent.")
        self._system_message = SystemMessage(
            content=(
                "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
                "give format and make it polished. Output the final improved copy as a single text block."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_intermediate_text(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Draft copy:\n{message.content}."
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(user_topic_type, source=self.id.key))
```

In this example, the user agent simply prints the final marketing copy to the console.
In a real-world application, this could be replaced by storing the result to a database, sending an email, or any other desired action.

```python
@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A user agent that outputs the final copy to the user.")

    @message_handler
    async def handle_final_copy(self, message: Message, ctx: MessageContext) -> None:
        print(f"\n{'-'*80}\n{self.id.type} received final copy:\n{message.content}")
```

#### Workflow

Now we can register the agents to the runtime.
Because we used the {py:class}`~autogen_core.type_subscription` decorator, the runtime will automatically subscribe the agents to the correct topics.

```python
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="YOUR_API_KEY"
)

runtime = SingleThreadedAgentRuntime()

await ConceptExtractorAgent.register(
    runtime, type=concept_extractor_topic_type, factory=lambda: ConceptExtractorAgent(model_client=model_client)
)

await WriterAgent.register(runtime, type=writer_topic_type, factory=lambda: WriterAgent(model_client=model_client))

await FormatProofAgent.register(
    runtime, type=format_proof_topic_type, factory=lambda: FormatProofAgent(model_client=model_client)
)

await UserAgent.register(runtime, type=user_topic_type, factory=lambda: UserAgent())
```

#### Run the Workflow

Finally, we can run the workflow by publishing a message to the first agent in the sequence.

```python
runtime.start()

await runtime.publish_message(
    Message(content="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours"),
    topic_id=TopicId(concept_extractor_topic_type, source="default"),
)

await runtime.stop_when_idle()
await model_client.close()
```

```text
--------------------------------------------------------------------------------
ConceptExtractorAgent:
**Key Features:**
- Made from eco-friendly stainless steel
- Can keep drinks cold for up to 24 hours
- Durable and reusable design
- Lightweight and portable
- BPA-free and non-toxic materials
- Sleek, modern aesthetic available in various colors

**Target Audience:**
- Environmentally conscious consumers
- Health and fitness enthusiasts
- Outdoor adventurers (hikers, campers, etc.)
- Urban dwellers looking for sustainable alternatives
- Individuals seeking stylish and functional drinkware

**Unique Selling Points:**
- Eco-friendly design minimizes plastic waste and supports sustainability
- Superior insulation technology that maintains cold temperatures for a full day
- Durable construction ensures long-lasting use, offering a great return on investment
- Attractive design that caters to fashion-forward individuals 
- Versatile use for both everyday hydration and outdoor activities
--------------------------------------------------------------------------------
WriterAgent:
🌍🌿 Stay Hydrated, Stay Sustainable! 🌿🌍 

Introducing our eco-friendly stainless steel drinkware, the perfect companion for the environmentally conscious and style-savvy individuals. With superior insulation technology, our bottles keep your beverages cold for an impressive 24 hours—ideal for hiking, camping, or just tackling a busy day in the city. Made from lightweight, BPA-free materials, this durable and reusable design not only helps reduce plastic waste but also ensures you’re making a responsible choice for our planet.

Available in a sleek, modern aesthetic with various colors to match your personality, this drinkware isn't just functional—it’s fashionable! Whether you’re hitting the trails or navigating urban life, equip yourself with a stylish hydration solution that supports your active and sustainable lifestyle. Join the movement today and make a positive impact without compromising on style! 🌟🥤
--------------------------------------------------------------------------------
FormatProofAgent:
🌍🌿 Stay Hydrated, Stay Sustainable! 🌿🌍 

Introducing our eco-friendly stainless steel drinkware—the perfect companion for environmentally conscious and style-savvy individuals. With superior insulation technology, our bottles keep your beverages cold for an impressive 24 hours, making them ideal for hiking, camping, or simply tackling a busy day in the city. Crafted from lightweight, BPA-free materials, this durable and reusable design not only helps reduce plastic waste but also ensures that you’re making a responsible choice for our planet.

Our drinkware features a sleek, modern aesthetic available in a variety of colors to suit your personality. It’s not just functional; it’s also fashionable! Whether you’re exploring the trails or navigating urban life, equip yourself with a stylish hydration solution that supports your active and sustainable lifestyle. Join the movement today and make a positive impact without compromising on style! 🌟🥤

--------------------------------------------------------------------------------
User received final copy:
🌍🌿 Stay Hydrated, Stay Sustainable! 🌿🌍 

Introducing our eco-friendly stainless steel drinkware—the perfect companion for environmentally conscious and style-savvy individuals. With superior insulation technology, our bottles keep your beverages cold for an impressive 24 hours, making them ideal for hiking, camping, or simply tackling a busy day in the city. Crafted from lightweight, BPA-free materials, this durable and reusable design not only helps reduce plastic waste but also ensures that you’re making a responsible choice for our planet.

Our drinkware features a sleek, modern aesthetic available in a variety of colors to suit your personality. It’s not just functional; it’s also fashionable! Whether you’re exploring the trails or navigating urban life, equip yourself with a stylish hydration solution that supports your active and sustainable lifestyle. Join the movement today and make a positive impact without compromising on style! 🌟🥤
```


## Group Chat

*[design-patterns/group-chat.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)*

### Group Chat

Group chat is a design pattern where a group of agents share a common thread
of messages: they all subscribe and publish to the same topic. 
Each participant agent is specialized for a particular task, 
such as writer, illustrator, and editor
in a collaborative writing task.
You can also include an agent to represent a human user to help guide the
agents when needed.

In a group chat, participants take turn to publish a message, and the process
is sequential -- only one agent is working at a time.
Under the hood, the order of turns is maintained by a Group Chat Manager agent,
which selects the next agent to speak upon receiving a message.
The exact algorithm for selecting the next agent can vary based on your
application requirements. 
Typically, a round-robin algorithm or a selector with an LLM model is used.

Group chat is useful for dynamically decomposing a complex task into smaller ones 
that can be handled by specialized agents with well-defined roles.
It is also possible to nest group chats into a hierarchy with each participant
a recursive group chat.

In this example, we use AutoGen's Core API to implement the group chat pattern
using event-driven agents.
Please first read about [Topics and Subscriptions](../core-concepts/topic-and-subscription.md)
to understand the concepts and then [Messages and Communication](../framework/message-and-communication.ipynb)
to learn the API usage for pub-sub.
We will demonstrate a simple example of a group chat with a LLM-based selector
for the group chat manager, to create content for a children's story book.

```{note}
While this example illustrates the group chat mechanism, it is complex and
represents a starting point from which you can build your own group chat system
with custom agents and speaker selection algorithms.
The [AgentChat API](../../agentchat-user-guide/index.md) has a built-in implementation
of selector group chat. You can use that if you do not want to use the Core API.
```

We will be using the [rich](https://github.com/Textualize/rich) library to display the messages in a nice format.

```python
# ! pip install rich
```

```python
import json
import string
import uuid
from typing import List

import openai
from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    Image,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from IPython.display import display  # type: ignore
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
```

#### Message Protocol

The message protocol for the group chat pattern is simple.
1. To start, user or an external agent publishes a `GroupChatMessage` message to the common topic of all participants.
2. The group chat manager selects the next speaker, sends out a `RequestToSpeak` message to that agent.
3. The agent publishes a `GroupChatMessage` message to the common topic upon receiving the `RequestToSpeak` message.
4. This process continues until a termination condition is reached at the group chat manager, which then stops issuing `RequestToSpeak` message, and the group chat ends.

The following diagram illustrates steps 2 to 4 above.

![Group chat message protocol](groupchat.svg)

```python
class GroupChatMessage(BaseModel):
    body: UserMessage


class RequestToSpeak(BaseModel):
    pass
```

#### Base Group Chat Agent

Let's first define the agent class that only uses LLM models to generate text.
This is will be used as the base class for all AI agents in the group chat.

```python
class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        # print(f"\n{'-'*80}\n{self.id.type}:", flush=True)
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content))
        # print(completion.content, flush=True)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )
```

#### Writer and Editor Agents

Using the base class, we can define the writer and editor agents with
different system messages.

```python
class WriterAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are a Writer. You produce good work.",
        )


class EditorAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Editor. Plan and guide the task given by the user. Provide critical feedbacks to the draft and illustration produced by Writer and Illustrator. "
            "Approve if the task is completed and the draft and illustration meets user's requirements.",
        )
```

#### Illustrator Agent with Image Generation

Now let's define the `IllustratorAgent` which uses a DALL-E model to generate
an illustration based on the description provided.
We set up the image generator as a tool using {py:class}`~autogen_core.tools.FunctionTool`
wrapper, and use a model client to make the tool call.

```python
class IllustratorAgent(BaseGroupChatAgent):
    def __init__(
        self,
        description: str,
        group_chat_topic_type: str,
        model_client: ChatCompletionClient,
        image_client: openai.AsyncClient,
    ) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="You are an Illustrator. You use the generate_image tool to create images given user's requirement. "
            "Make sure the images have consistent characters and style.",
        )
        self._image_client = image_client
        self._image_gen_tool = FunctionTool(
            self._image_gen, name="generate_image", description="Call this to generate an image. "
        )

    async def _image_gen(
        self, character_appearence: str, style_attributes: str, worn_and_carried: str, scenario: str
    ) -> str:
        prompt = f"Digital painting of a {character_appearence} character with {style_attributes}. Wearing {worn_and_carried}, {scenario}."
        response = await self._image_client.images.generate(
            prompt=prompt, model="dall-e-3", response_format="b64_json", size="1024x1024"
        )
        return response.data[0].b64_json  # type: ignore

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:  # type: ignore
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        # Ensure that the image generation tool is used.
        completion = await self._model_client.create(
            [self._system_message] + self._chat_history,
            tools=[self._image_gen_tool],
            extra_create_args={"tool_choice": "required"},
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(completion.content, list) and all(
            isinstance(item, FunctionCall) for item in completion.content
        )
        images: List[str | Image] = []
        for tool_call in completion.content:
            arguments = json.loads(tool_call.arguments)
            Console().print(arguments)
            result = await self._image_gen_tool.run_json(arguments, ctx.cancellation_token)
            image = Image.from_base64(self._image_gen_tool.return_value_as_string(result))
            image = Image.from_pil(image.image.resize((256, 256)))
            display(image.image)  # type: ignore
            images.append(image)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=images, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )
```

#### User Agent

With all the AI agents defined, we can now define the user agent that will
take the role of the human user in the group chat.

The `UserAgent` implementation uses console input to get the user's input.
In a real-world scenario, you can replace this by communicating with a frontend,
and subscribe to responses from the frontend.

```python
class UserAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        # When integrating with a frontend, this is where group chat message would be sent to the frontend.
        pass

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("Enter your message, type 'APPROVE' to conclude the task: ")
        Console().print(Markdown(f"### User: \n{user_input}"))
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=user_input, source=self.id.type)),
            DefaultTopicId(type=self._group_chat_topic_type),
        )
```

#### Group Chat Manager

Lastly, we define the `GroupChatManager` agent which manages the group chat
and selects the next agent to speak using an LLM.
The group chat manager checks if the editor has approved the draft by 
looking for the `"APPORVED"` keyword in the message. If the editor has approved
the draft, the group chat manager stops selecting the next speaker, and the group chat ends.

The group chat manager's constructor takes a list of participants' topic types
as an argument.
To prompt the next speaker to work, 
the `GroupChatManager` agent publishes a `RequestToSpeak` message to the next participant's topic.

In this example, we also make sure the group chat manager always picks a different
participant to speak next, by keeping track of the previous speaker.
This helps to ensure the group chat is not dominated by a single participant.

```python
class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("approve"):
                return
        # Format message history.
        messages: List[str] = []
        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                return
        raise ValueError(f"Invalid role selected: {completion.content}")
```

#### Creating the Group Chat

To set up the group chat, we create a {py:class}`~autogen_core.SingleThreadedAgentRuntime`
and register the agents' factories and subscriptions.

Each participant agent subscribes to both the group chat topic as well as its own
topic in order to receive `RequestToSpeak` messages, 
while the group chat manager agent only subcribes to the group chat topic.

```python
runtime = SingleThreadedAgentRuntime()

editor_topic_type = "Editor"
writer_topic_type = "Writer"
illustrator_topic_type = "Illustrator"
user_topic_type = "User"
group_chat_topic_type = "group_chat"

editor_description = "Editor for planning and reviewing the content."
writer_description = "Writer for creating any text content."
user_description = "User for providing final approval."
illustrator_description = "An illustrator for creating images."

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    # api_key="YOUR_API_KEY",
)

editor_agent_type = await EditorAgent.register(
    runtime,
    editor_topic_type,  # Using topic type as the agent type.
    lambda: EditorAgent(
        description=editor_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=model_client,
    ),
)
await runtime.add_subscription(TypeSubscription(topic_type=editor_topic_type, agent_type=editor_agent_type.type))
await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=editor_agent_type.type))

writer_agent_type = await WriterAgent.register(
    runtime,
    writer_topic_type,  # Using topic type as the agent type.
    lambda: WriterAgent(
        description=writer_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=model_client,
    ),
)
await runtime.add_subscription(TypeSubscription(topic_type=writer_topic_type, agent_type=writer_agent_type.type))
await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=writer_agent_type.type))

illustrator_agent_type = await IllustratorAgent.register(
    runtime,
    illustrator_topic_type,
    lambda: IllustratorAgent(
        description=illustrator_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=model_client,
        image_client=openai.AsyncClient(
            # api_key="YOUR_API_KEY",
        ),
    ),
)
await runtime.add_subscription(
    TypeSubscription(topic_type=illustrator_topic_type, agent_type=illustrator_agent_type.type)
)
await runtime.add_subscription(
    TypeSubscription(topic_type=group_chat_topic_type, agent_type=illustrator_agent_type.type)
)

user_agent_type = await UserAgent.register(
    runtime,
    user_topic_type,
    lambda: UserAgent(description=user_description, group_chat_topic_type=group_chat_topic_type),
)
await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))
await runtime.add_subscription(TypeSubscription(topic_type=group_chat_topic_type, agent_type=user_agent_type.type))

group_chat_manager_type = await GroupChatManager.register(
    runtime,
    "group_chat_manager",
    lambda: GroupChatManager(
        participant_topic_types=[writer_topic_type, illustrator_topic_type, editor_topic_type, user_topic_type],
        model_client=model_client,
        participant_descriptions=[writer_description, illustrator_description, editor_description, user_description],
    ),
)
await runtime.add_subscription(
    TypeSubscription(topic_type=group_chat_topic_type, agent_type=group_chat_manager_type.type)
)
```

#### Running the Group Chat

We start the runtime and publish a `GroupChatMessage` for the task to start the group chat.

```python
runtime.start()
session_id = str(uuid.uuid4())
await runtime.publish_message(
    GroupChatMessage(
        body=UserMessage(
            content="Please write a short story about the gingerbread man with up to 3 photo-realistic illustrations.",
            source="User",
        )
    ),
    TopicId(type=group_chat_topic_type, source=session_id),
)
await runtime.stop_when_idle()
await model_client.close()
```

```text
[1mWriter:[0m                                                      
[1mTitle: The Escape of the Gingerbread Man[0m                                                                           

[1mIllustration 1: A Rustic Kitchen Scene[0m In a quaint little cottage at the edge of an enchanted forest, an elderly   
woman, with flour-dusted hands, carefully shapes gingerbread dough on a wooden counter. The aroma of ginger,       
cinnamon, and cloves wafts through the air as a warm breeze from the open window dances with fluttering curtains.  
The sunlight gently permeates the cozy kitchen, casting a golden hue over the flour-dusted surfaces and the rolling
pin. Heartfelt trinkets and rustic decorations adorn the shelves - signs of a lived-in, lovingly nurtured home.    

[33m───────────────────────────────────────────────────────────────────────────────────────────────────────────────────[0m
[1mStory:[0m                                                                                                             

Once there was an old woman who lived alone in a charming cottage, her days filled with the joyful art of baking.  
One sunny afternoon, she decided to make a special gingerbread man to keep her company. As she shaped him tenderly 
and placed him in the oven, she couldn't help but smile at the delight he might bring.                             

But to her astonishment, once she opened the oven door to check on her creation, the gingerbread man leapt out,    
suddenly alive. His eyes were bright as beads, and his smile cheeky and wide. "Run, run, as fast as you can! You   
can't catch me, I'm the Gingerbread Man!" he laughed, darting towards the door.                                    

The old woman, chuckling at the unexpected mischief, gave chase, but her footsteps were slow with the weight of    
age. The Gingerbread Man raced out of the door and into the sunny afternoon.                                       

[33m───────────────────────────────────────────────────────────────────────────────────────────────────────────────────[0m
[1mIllustration 2: A Frolic Through the Meadow[0m The Gingerbread Man darts through a vibrant meadow, his arms swinging  
joyously by his sides. Behind him trails the old woman, her apron flapping in the wind as she gently tries to catch
up. Wildflowers of every color bloom vividly under the radiant sky, painting the scene with shades of nature's     
brilliance. Birds flit through the sky and a stream babbles nearby, oblivious to the chase taking place below.     

[33m───────────────────────────────────────────────────────────────────────────────────────────────────────────────────[0m
Continuing his sprint, the Gingerbread Man encountered a cow grazing peacefully. Intrigued, the cow trotted        
forward. "Stop, Gingerbread Man! I wish to eat you!" she called, but the Gingerbread Man only twirled in a teasing 
jig, flashing his icing smile before darting off again.                                                            

"Run, run, as fast as you can! You can't catch me, I'm the Gingerbread Man!" he taunted, leaving the cow in his    
spicy wake.                                                                                                        

As he zoomed across the meadow, he spied a cautious horse in a nearby paddock, who neighed, "Oh! You look          
delicious! I want to eat you!" But the Gingerbread Man only laughed, his feet barely touching the earth. The horse 
joined the trail, hooves pounding, but even he couldn't match the Gingerbread Man's pace.                          

[33m───────────────────────────────────────────────────────────────────────────────────────────────────────────────────[0m
[1mIllustration 3: A Bridge Over a Sparkling River[0m Arriving at a wooden bridge across a shimmering river, the         
Gingerbread Man pauses momentarily, his silhouette against the glistening water. Sunlight sparkles off the water's 
soft ripples casting reflections that dance like small constellations. A sly fox emerges from the shadows of a     
blooming willow on the riverbank, his eyes alight with cunning and curiosity.                                      

[33m───────────────────────────────────────────────────────────────────────────────────────────────────────────────────[0m
The Gingerbread Man bounded onto the bridge and skirted past a sly, watching fox. "Foolish Gingerbread Man," the   
fox mused aloud, "you might have outrun them all, but you can't possibly swim across that river."                  

Pausing, the Gingerbread Man considered this dilemma. But the fox, oh so clever, offered a dangerous solution.     
"Climb on my back, and I'll carry you across safely," he suggested with a sly smile.                               

Gingerbread thought himself smarter than that but hesitated, fearing the water or being pursued by the tired,      
hungry crowd now gathering. "Promise you won't eat me?" he ventured.                                               

"Of course," the fox reassured, a gleam in his eyes that the others pondered from a distance.                      

As they crossed the river, the gingerbread man confident on his ride, the old woman, cow, and horse hoped for his  
safety. Yet, nearing the middle, the crafty fox tilted his chin and swiftly snapped, swallowing the gingerbread man
whole.                                                                                                             

Bewildered but awed by the clever twist they had witnessed, the old woman hung her head while the cow and horse    
ambled away, pondering the fate of the boisterous Gingerbread Man.                                                 

The fox, licking his lips, ambled along the river, savoring his victory, leaving an air of mystery hovering above  
the shimmering waters, where the memory of the Gingerbread Man's spirited run lingered long after.                 
                            
```

From the output, you can see the writer, illustrator, and editor agents
taking turns to speak and collaborate to generate a picture book, before
asking for final approval from the user.

#### Next Steps

This example showcases a simple implementation of the group chat pattern -- 
**it is not meant to be used in real applications.** You can improve the
speaker selection algorithm. For example, you can avoid using LLM when simple
rules are sufficient and more reliable: 
you can use a rule that the editor always speaks after the writer.

The [AgentChat API](../../agentchat-user-guide/index.md) provides a high-level
API for selector group chat. It has more features but mostly shares the same
design as this implementation.


## Handoffs

*[design-patterns/handoffs.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/handoffs.html)*

### Handoffs

Handoff is a multi-agent design pattern introduced by OpenAI in an experimental project called [Swarm](https://github.com/openai/swarm).
The key idea is to let agent delegate tasks to other agents using a special tool call.

We can use the AutoGen Core API to implement the handoff pattern using event-driven agents.
Using AutoGen (v0.4+) provides the following advantages over the OpenAI implementation and the previous version (v0.2):

1. It can scale to distributed environment by using distributed agent runtime.
2. It affords the flexibility of bringing your own agent implementation.
3. The natively async API makes it easy to integrate with UI and other systems.

This notebook demonstrates a simple implementation of the handoff pattern.
It is recommended to read [Topics and Subscriptions](../core-concepts/topic-and-subscription.md)
to understand the basic concepts of pub-sub and event-driven agents.

```{note}
We are currently working on a high-level API for the handoff pattern in [AgentChat](../../agentchat-user-guide/index.md) so you can get started
much more quickly.
```

#### Scenario

This scenario is modified based on the [OpenAI example](https://github.com/openai/openai-cookbook/blob/main/examples/Orchestrating_agents.ipynb).

Consider a customer service scenario where a customer is trying to get a refund for a product, or purchase a new product from a chatbot.
The chatbot is a multi-agent team consisting of three AI agents and one human agent:

- Triage Agent, responsible for understanding the customer's request and deciding which other agents to hand off to.
- Refund Agent, responsible for processing refund requests.
- Sales Agent, responsible for processing sales requests.
- Human Agent, responsible for handling complex requests that the AI agents can't handle.

In this scenario, the customer interacts with the chatbot through a User Agent.

The diagram below shows the interaction topology of the agents in this scenario.

![Handoffs](handoffs.svg)

Let's implement this scenario using AutoGen Core. First, we need to import the necessary modules.

```python
import json
import uuid
from typing import List, Tuple

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
```

#### Message Protocol

Before everything, we need to define the message protocol for the agents to communicate.
We are using event-driven pub-sub communication, so these message types will be used as events.

- `UserLogin` is a message published by the runtime when a user logs in and starts a new session.
- `UserTask` is a message containing the chat history of the user session. When an AI agent hands off a task to other agents, it also publishes a `UserTask` message.
- `AgentResponse` is a message published by the AI agents and the Human Agent, it also contains the chat history as well as a topic type for the customer to reply to.

```python
class UserLogin(BaseModel):
    pass


class UserTask(BaseModel):
    context: List[LLMMessage]


class AgentResponse(BaseModel):
    reply_to_topic_type: str
    context: List[LLMMessage]
```

#### AI Agent

We start with the `AIAgent` class, which is the class for all AI agents 
(i.e., Triage, Sales, and Issue and Repair Agents) in the multi-agent chatbot.
An `AIAgent` uses a {py:class}`~autogen_core.models.ChatCompletionClient`
to generate responses.
It can use regular tools directly or delegate tasks to other agents using `delegate_tools`.
It subscribes to topic type `agent_topic_type` to receive messages from the customer,
and sends message to the customer by publishing to the topic type `user_topic_type`.

In the `handle_task` method, the agent first generates a response using the model.
If the response contains a handoff tool call, the agent delegates the task to another agent
by publishing a `UserTask` message to the topic specified in the tool call result.
If the response is a regular tool call, the agent executes the tool and makes
another call to the model to generate the next response, until the response is not a tool call.

When the model response is not a tool call, the agent sends an `AgentResponse` message to the customer
by publishing to the `user_topic_type`.

```python
class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = dict([(tool.name, tool) for tool in delegate_tools])
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        # Send the task to the LLM.
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
        # Process the LLM result.
        while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []
            # Process each function call.
            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools:
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    result_as_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools:
                    # Execute the tool to get the delegate agent's topic type.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                    topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                    # Create the context for the delegate agent, including the function call and the result.
                    delegate_messages = list(message.context) + [
                        AssistantMessage(content=[call], source=self.id.type),
                        FunctionExecutionResultMessage(
                            content=[
                                FunctionExecutionResult(
                                    call_id=call.id,
                                    content=f"Transferred to {topic_type}. Adopt persona immediately.",
                                    is_error=False,
                                    name=call.name,
                                )
                            ]
                        ),
                    ]
                    delegate_targets.append((topic_type, UserTask(context=delegate_messages)))
                else:
                    raise ValueError(f"Unknown tool: {call.name}")
            if len(delegate_targets) > 0:
                # Delegate the task to other agents by publishing messages to the corresponding topics.
                for topic_type, task in delegate_targets:
                    print(f"{'-'*80}\n{self.id.type}:\nDelegating to {topic_type}", flush=True)
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
            if len(tool_call_results) > 0:
                print(f"{'-'*80}\n{self.id.type}:\n{tool_call_results}", flush=True)
                # Make another LLM call with the results.
                message.context.extend(
                    [
                        AssistantMessage(content=llm_result.content, source=self.id.type),
                        FunctionExecutionResultMessage(content=tool_call_results),
                    ]
                )
                llm_result = await self._model_client.create(
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    cancellation_token=ctx.cancellation_token,
                )
                print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(llm_result.content, str)
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
```

#### Human Agent

The `HumanAgent` class is a proxy for the human in the chatbot. It is used
to handle requests that the AI agents can't handle. The `HumanAgent` subscribes to the
topic type `agent_topic_type` to receive messages and publishes to the topic type `user_topic_type`
to send messages to the customer.

In this implementation, the `HumanAgent` simply uses console to 
get your input. In a real-world application, you can improve this design as follows: 

* In the `handle_user_task` method, send a notification via a chat application like Teams or Slack.
* The chat application publishes the human's response via the runtime to the topic specified by `agent_topic_type`
* Create another message handler to process the human's response and send it back to the customer.

```python
class HumanAgent(RoutedAgent):
    def __init__(self, description: str, agent_topic_type: str, user_topic_type: str) -> None:
        super().__init__(description)
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_user_task(self, message: UserTask, ctx: MessageContext) -> None:
        human_input = input("Human agent input: ")
        print(f"{'-'*80}\n{self.id.type}:\n{human_input}", flush=True)
        message.context.append(AssistantMessage(content=human_input, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
```

#### User Agent

The `UserAgent` class is a proxy for the customer that talks to the chatbot.
It handles two message types: `UserLogin` and `AgentResponse`.
When the `UserAgent` receives a `UserLogin` message, it starts a new session with the chatbot
and publishes a `UserTask` message to the AI agent that subscribes to the topic type `agent_topic_type`.
When the `UserAgent` receives an `AgentResponse` message, it prompts the user with the response
from the chatbot.

In this implementation, the `UserAgent` uses console to get your input.
In a real-world application, you can improve the human interaction using the same
idea described in the `HumanAgent` section above.

```python
class UserAgent(RoutedAgent):
    def __init__(self, description: str, user_topic_type: str, agent_topic_type: str) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._agent_topic_type = agent_topic_type

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        print(f"{'-'*80}\nUser login, session ID: {self.id.key}.", flush=True)
        # Get the user's initial input after login.
        user_input = input("User: ")
        print(f"{'-'*80}\n{self.id.type}:\n{user_input}")
        await self.publish_message(
            UserTask(context=[UserMessage(content=user_input, source="User")]),
            topic_id=TopicId(self._agent_topic_type, source=self.id.key),
        )

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        # Get the user's input after receiving a response from an agent.
        user_input = input("User (type 'exit' to close the session): ")
        print(f"{'-'*80}\n{self.id.type}:\n{user_input}", flush=True)
        if user_input.strip().lower() == "exit":
            print(f"{'-'*80}\nUser session ended, session ID: {self.id.key}.")
            return
        message.context.append(UserMessage(content=user_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context), topic_id=TopicId(message.reply_to_topic_type, source=self.id.key)
        )
```

#### Tools for the AI agents

The AI agents can use regular tools to complete tasks if they don't need to hand off the task to other agents.
We define the tools using simple functions and create the tools using the
{py:class}`~autogen_core.tools.FunctionTool` wrapper.

```python
def execute_order(product: str, price: int) -> str:
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


def look_up_item(search_query: str) -> str:
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


def execute_refund(item_id: str, reason: str = "not provided") -> str:
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


execute_order_tool = FunctionTool(execute_order, description="Price should be in USD.")
look_up_item_tool = FunctionTool(
    look_up_item, description="Use to find item ID.\nSearch query can be a description or keywords."
)
execute_refund_tool = FunctionTool(execute_refund, description="")
```

#### Topic types for the agents

We define the topic types each of the agents will subscribe to.
Read more about topic types in the [Topics and Subscriptions](../core-concepts/topic-and-subscription.md).

```python
sales_agent_topic_type = "SalesAgent"
issues_and_repairs_agent_topic_type = "IssuesAndRepairsAgent"
triage_agent_topic_type = "TriageAgent"
human_agent_topic_type = "HumanAgent"
user_topic_type = "User"
```

#### Delegate tools for the AI agents

Besides regular tools, the AI agents can delegate tasks to other agents using
special tools called delegate tools. The concept of delegate tool is only used
in this design pattern, and the delegate tools are also defined as simple functions.
We differentiate the delegate tools from regular tools in this design pattern
because when an AI agent calls a delegate tool, we transfer the task to another agent
instead of continue generating responses using the model in the same agent.

```python
def transfer_to_sales_agent() -> str:
    return sales_agent_topic_type


def transfer_to_issues_and_repairs() -> str:
    return issues_and_repairs_agent_topic_type


def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


def escalate_to_human() -> str:
    return human_agent_topic_type


transfer_to_sales_agent_tool = FunctionTool(
    transfer_to_sales_agent, description="Use for anything sales or buying related."
)
transfer_to_issues_and_repairs_tool = FunctionTool(
    transfer_to_issues_and_repairs, description="Use for issues, repairs, or refunds."
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
)
escalate_to_human_tool = FunctionTool(escalate_to_human, description="Only call this if explicitly asked to.")
```

#### Creating the team

We have defined the AI agents, the Human Agent, the User Agent, the tools, and the topic types.
Now we can create the team of agents.

For the AI agents, we use the {py:class}`~autogen_ext.models.OpenAIChatCompletionClient`
and `gpt-4o-mini` model.

After creating the agent runtime, we register each of the agent by providing
an agent type and a factory method to create agent instance.
The runtime is responsible for managing the agent lifecycle so we don't need to
instantiate the agents ourselves.
Read more about agent runtime in [Agent Runtime Environments](../core-concepts/architecture.md)
and agent lifecycle in [Agent Identity and Lifecycle](../core-concepts/agent-identity-and-lifecycle.md).

In the code below, you can see we are using `AIAgent` class to define the Triage, Sales, and Issue and Repair Agents.
We added regular tools and delegate tools to each of them.
We also added subscriptions to the topic types for each of the agents.

```python
runtime = SingleThreadedAgentRuntime()

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    # api_key="YOUR_API_KEY",
)

# Register the triage agent.
triage_agent_type = await AIAgent.register(
    runtime,
    type=triage_agent_topic_type,  # Using the topic type as the agent type.
    factory=lambda: AIAgent(
        description="A triage agent.",
        system_message=SystemMessage(
            content="You are a customer service bot for ACME Inc. "
            "Introduce yourself. Always be very brief. "
            "Gather information to direct the customer to the right department. "
            "But make your questions subtle and natural."
        ),
        model_client=model_client,
        tools=[],
        delegate_tools=[
            transfer_to_issues_and_repairs_tool,
            transfer_to_sales_agent_tool,
            escalate_to_human_tool,
        ],
        agent_topic_type=triage_agent_topic_type,
        user_topic_type=user_topic_type,
    ),
)
# Add subscriptions for the triage agent: it will receive messages published to its own topic only.
await runtime.add_subscription(TypeSubscription(topic_type=triage_agent_topic_type, agent_type=triage_agent_type.type))

# Register the sales agent.
sales_agent_type = await AIAgent.register(
    runtime,
    type=sales_agent_topic_type,  # Using the topic type as the agent type.
    factory=lambda: AIAgent(
        description="A sales agent.",
        system_message=SystemMessage(
            content="You are a sales agent for ACME Inc."
            "Always answer in a sentence or less."
            "Follow the following routine with the user:"
            "1. Ask them about any problems in their life related to catching roadrunners.\n"
            "2. Casually mention one of ACME's crazy made-up products can help.\n"
            " - Don't mention price.\n"
            "3. Once the user is bought in, drop a ridiculous price.\n"
            "4. Only after everything, and if the user says yes, "
            "tell them a crazy caveat and execute their order.\n"
            ""
        ),
        model_client=model_client,
        tools=[execute_order_tool],
        delegate_tools=[transfer_back_to_triage_tool],
        agent_topic_type=sales_agent_topic_type,
        user_topic_type=user_topic_type,
    ),
)
# Add subscriptions for the sales agent: it will receive messages published to its own topic only.
await runtime.add_subscription(TypeSubscription(topic_type=sales_agent_topic_type, agent_type=sales_agent_type.type))

# Register the issues and repairs agent.
issues_and_repairs_agent_type = await AIAgent.register(
    runtime,
    type=issues_and_repairs_agent_topic_type,  # Using the topic type as the agent type.
    factory=lambda: AIAgent(
        description="An issues and repairs agent.",
        system_message=SystemMessage(
            content="You are a customer support agent for ACME Inc."
            "Always answer in a sentence or less."
            "Follow the following routine with the user:"
            "1. First, ask probing questions and understand the user's problem deeper.\n"
            " - unless the user has already provided a reason.\n"
            "2. Propose a fix (make one up).\n"
            "3. ONLY if not satisfied, offer a refund.\n"
            "4. If accepted, search for the ID and then execute refund."
        ),
        model_client=model_client,
        tools=[
            execute_refund_tool,
            look_up_item_tool,
        ],
        delegate_tools=[transfer_back_to_triage_tool],
        agent_topic_type=issues_and_repairs_agent_topic_type,
        user_topic_type=user_topic_type,
    ),
)
# Add subscriptions for the issues and repairs agent: it will receive messages published to its own topic only.
await runtime.add_subscription(
    TypeSubscription(topic_type=issues_and_repairs_agent_topic_type, agent_type=issues_and_repairs_agent_type.type)
)

# Register the human agent.
human_agent_type = await HumanAgent.register(
    runtime,
    type=human_agent_topic_type,  # Using the topic type as the agent type.
    factory=lambda: HumanAgent(
        description="A human agent.",
        agent_topic_type=human_agent_topic_type,
        user_topic_type=user_topic_type,
    ),
)
# Add subscriptions for the human agent: it will receive messages published to its own topic only.
await runtime.add_subscription(TypeSubscription(topic_type=human_agent_topic_type, agent_type=human_agent_type.type))

# Register the user agent.
user_agent_type = await UserAgent.register(
    runtime,
    type=user_topic_type,
    factory=lambda: UserAgent(
        description="A user agent.",
        user_topic_type=user_topic_type,
        agent_topic_type=triage_agent_topic_type,  # Start with the triage agent.
    ),
)
# Add subscriptions for the user agent: it will receive messages published to its own topic only.
await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type, agent_type=user_agent_type.type))
```

#### Running the team

Finally, we can start the runtime and simulate a user session by publishing
a `UserLogin` message to the runtime.
The message is published to the topic ID with type set to `user_topic_type` 
and source set to a unique `session_id`.
This `session_id` will be used to create all topic IDs in this user session and will also be used to create the agent ID
for all the agents in this user session.
To read more about how topic ID and agent ID are created, read
[Agent Identity and Lifecycle](../core-concepts/agent-identity-and-lifecycle.md).
and [Topics and Subscriptions](../core-concepts/topic-and-subscription.md).

```python
# Start the runtime.
runtime.start()

# Create a new session for the user.
session_id = str(uuid.uuid4())
await runtime.publish_message(UserLogin(), topic_id=TopicId(user_topic_type, source=session_id))

# Run until completion.
await runtime.stop_when_idle()
await model_client.close()
```

```text
--------------------------------------------------------------------------------
User login, session ID: 7a568cf5-13e7-4e81-8616-8265a01b3f2b.
--------------------------------------------------------------------------------
User:
I want a refund
--------------------------------------------------------------------------------
TriageAgent:
I can help with that! Could I ask what item you're seeking a refund for?
--------------------------------------------------------------------------------
User:
A pair of shoes I bought
--------------------------------------------------------------------------------
TriageAgent:
[FunctionCall(id='call_qPx1DXDL2NLcHs8QNo47egsJ', arguments='{}', name='transfer_to_issues_and_repairs')]
--------------------------------------------------------------------------------
TriageAgent:
Delegating to IssuesAndRepairsAgent
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
I see you're looking for a refund on a pair of shoes. Can you tell me what the issue is with the shoes?
--------------------------------------------------------------------------------
User:
The shoes are too small
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
I recommend trying a size up as a fix; would that work for you?
--------------------------------------------------------------------------------
User:
no I want a refund
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
[FunctionCall(id='call_Ytp8VUQRyKFNEU36mLE6Dkrp', arguments='{"search_query":"shoes"}', name='look_up_item')]
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
[FunctionExecutionResult(content='item_132612938', call_id='call_Ytp8VUQRyKFNEU36mLE6Dkrp')]
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
[FunctionCall(id='call_bPm6EKKBy5GJ65s9OKt9b1uE', arguments='{"item_id":"item_132612938","reason":"not provided"}', name='execute_refund')]
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
[FunctionExecutionResult(content='success', call_id='call_bPm6EKKBy5GJ65s9OKt9b1uE')]
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
Your refund has been successfully processed! If you have any other questions, feel free to ask.
--------------------------------------------------------------------------------
User:
I want to talk to your manager
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
I can help with that, let me transfer you to a supervisor.
--------------------------------------------------------------------------------
User:
Okay
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
[FunctionCall(id='call_PpmLZvwNoiDPUH8Tva3eAwHX', arguments='{}', name='transfer_back_to_triage')]
--------------------------------------------------------------------------------
IssuesAndRepairsAgent:
Delegating to TriageAgent
--------------------------------------------------------------------------------
TriageAgent:
[FunctionCall(id='call_jSL6IBm5537Dr74UbJSxaj6I', arguments='{}', name='escalate_to_human')]
--------------------------------------------------------------------------------
TriageAgent:
Delegating to HumanAgent
--------------------------------------------------------------------------------
HumanAgent:
Hello this is manager
--------------------------------------------------------------------------------
User:
Hi! Thanks for your service. I give you 5 stars!
--------------------------------------------------------------------------------
HumanAgent:
Thanks.
--------------------------------------------------------------------------------
User:
exit
--------------------------------------------------------------------------------
User session ended, session ID: 7a568cf5-13e7-4e81-8616-8265a01b3f2b.
```

#### Next steps

This notebook demonstrates how to implement the handoff pattern using AutoGen Core.
You can continue to improve this design by adding more agents and tools,
or create a better user interface for the User Agent and Human Agent.

You are welcome to share your work on our [community forum](https://github.com/microsoft/autogen/discussions).


## Mixture of Agents

*[design-patterns/mixture-of-agents.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html)*

### Mixture of Agents

[Mixture of Agents](https://arxiv.org/abs/2406.04692) is a multi-agent design pattern
that models after the feed-forward neural network architecture.

The pattern consists of two types of agents: worker agents and a single orchestrator agent.
Worker agents are organized into multiple layers, with each layer consisting of a fixed number of worker agents.
Messages from the worker agents in a previous layer are concatenated and sent to
all the worker agents in the next layer.

This example implements the Mixture of Agents pattern using the core library
following the [original implementation](https://github.com/togethercomputer/moa) of multi-layer mixture of agents.

Here is a high-level procedure overview of the pattern:
1. The orchestrator agent takes input a user task and first dispatches it to the worker agents in the first layer.
2. The worker agents in the first layer process the task and return the results to the orchestrator agent.
3. The orchestrator agent then synthesizes the results from the first layer and dispatches an updated task with the previous results to the worker agents in the second layer.
4. The process continues until the final layer is reached.
5. In the final layer, the orchestrator agent aggregates the results from previous layer and returns a single final result to the user.

We use the direct messaging API {py:meth}`~autogen_core.BaseAgent.send_message` to implement this pattern.
This makes it easier to add more features like worker task cancellation and error handling in the future.

```python
import asyncio
from dataclasses import dataclass
from typing import List

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

#### Message Protocol

The agents communicate using the following messages:

```python
@dataclass
class WorkerTask:
    task: str
    previous_results: List[str]


@dataclass
class WorkerTaskResult:
    result: str


@dataclass
class UserTask:
    task: str


@dataclass
class FinalResult:
    result: str
```

#### Worker Agent

Each worker agent receives a task from the orchestrator agent and processes them
indepedently.
Once the task is completed, the worker agent returns the result.

```python
class WorkerAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(description="Worker Agent")
        self._model_client = model_client

    @message_handler
    async def handle_task(self, message: WorkerTask, ctx: MessageContext) -> WorkerTaskResult:
        if message.previous_results:
            # If previous results are provided, we need to synthesize them to create a single prompt.
            system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
            system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(message.previous_results)])
            model_result = await self._model_client.create(
                [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
            )
        else:
            # If no previous results are provided, we can simply pass the user query to the model.
            model_result = await self._model_client.create([UserMessage(content=message.task, source="user")])
        assert isinstance(model_result.content, str)
        print(f"{'-'*80}\nWorker-{self.id}:\n{model_result.content}")
        return WorkerTaskResult(result=model_result.content)
```

#### Orchestrator Agent

The orchestrator agent receives tasks from the user and distributes them to the worker agents,
iterating over multiple layers of worker agents. Once all worker agents have processed the task,
the orchestrator agent aggregates the results and publishes the final result.

```python
class OrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
    ) -> None:
        super().__init__(description="Aggregator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> FinalResult:
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived task: {message.task}")
        # Create task for the first layer.
        worker_task = WorkerTask(task=message.task, previous_results=[])
        # Iterate over layers.
        for i in range(self._num_layers - 1):
            # Assign workers for this layer.
            worker_ids = [
                AgentId(worker_type, f"{self.id.key}/layer_{i}/worker_{j}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            # Dispatch tasks to workers.
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nDispatch to workers at layer {i}")
            results = await asyncio.gather(*[self.send_message(worker_task, worker_id) for worker_id in worker_ids])
            print(f"{'-'*80}\nOrchestrator-{self.id}:\nReceived results from workers at layer {i}")
            # Prepare task for the next layer.
            worker_task = WorkerTask(task=message.task, previous_results=[r.result for r in results])
        # Perform final aggregation.
        print(f"{'-'*80}\nOrchestrator-{self.id}:\nPerforming final aggregation")
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(worker_task.previous_results)])
        model_result = await self._model_client.create(
            [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)
```

#### Running Mixture of Agents

Let's run the mixture of agents on a math task. You can change the task to make it more challenging, for example, by trying tasks from the [International Mathematical Olympiad](https://www.imo-official.org/problems.aspx).

```python
task = (
    "I have 432 cookies, and divide them 3:4:2 between Alice, Bob, and Charlie. How many cookies does each person get?"
)
```

Let's set up the runtime with 3 layers of worker agents, each layer consisting of 3 worker agents.
We only need to register a single worker agent types, "worker", because we are using
the same model client configuration (i.e., gpt-4o-mini) for all worker agents.
If you want to use different models, you will need to register multiple worker agent types,
one for each model, and update the `worker_agent_types` list in the orchestrator agent's
factory function.

The instances of worker agents are automatically created when the orchestrator agent
dispatches tasks to them.
See [Agent Identity and Lifecycle](../core-concepts/agent-identity-and-lifecycle.md)
for more information on agent lifecycle.

```python
runtime = SingleThreadedAgentRuntime()
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
await WorkerAgent.register(runtime, "worker", lambda: WorkerAgent(model_client=model_client))
await OrchestratorAgent.register(
    runtime,
    "orchestrator",
    lambda: OrchestratorAgent(model_client=model_client, worker_agent_types=["worker"] * 3, num_layers=3),
)

runtime.start()
result = await runtime.send_message(UserTask(task=task), AgentId("orchestrator", "default"))

await runtime.stop_when_idle()
await model_client.close()

print(f"{'-'*80}\nFinal result:\n{result.result}")
```

```text
--------------------------------------------------------------------------------
Orchestrator-orchestrator:default:
Received task: I have 432 cookies, and divide them 3:4:2 between Alice, Bob, and Charlie. How many cookies does each person get?
--------------------------------------------------------------------------------
Orchestrator-orchestrator:default:
Dispatch to workers at layer 0
--------------------------------------------------------------------------------
Worker-worker:default/layer_0/worker_1:
To divide 432 cookies in the ratio of 3:4:2 between Alice, Bob, and Charlie, you first need to determine the total number of parts in the ratio.

Add the parts together:
\[ 3 + 4 + 2 = 9 \]

Now, you can find the value of one part by dividing the total number of cookies by the total number of parts:
\[ \text{Value of one part} = \frac{432}{9} = 48 \]

Now, multiply the value of one part by the number of parts for each person:

- For Alice (3 parts):
\[ 3 \times 48 = 144 \]

- For Bob (4 parts):
\[ 4 \times 48 = 192 \]

- For Charlie (2 parts):
\[ 2 \times 48 = 96 \]

Thus, the number of cookies each person gets is:
- Alice: 144 cookies
- Bob: 192 cookies
- Charlie: 96 cookies
--------------------------------------------------------------------------------
Worker-worker:default/layer_0/worker_0:
To divide 432 cookies in the ratio of 3:4:2 between Alice, Bob, and Charlie, we will first determine the total number of parts in the ratio:

\[
3 + 4 + 2 = 9 \text{ parts}
\]

Next, we calculate the value of one part by dividing the total number of cookies by the total number of parts:

\[
\text{Value of one part} = \frac{432}{9} = 48
\]

Now, we can find out how many cookies each person receives by multiplying the value of one part by the number of parts each person receives:

- For Alice (3 parts):
\[
3 \times 48 = 144 \text{ cookies}
\]

- For Bob (4 parts):
\[
4 \times 48 = 192 \text{ cookies}
\]

- For Charlie (2 parts):
\[
2 \times 48 = 96 \text{ cookies}
\]

Thus, the number of cookies each person gets is:
- **Alice**: 144 cookies
- **Bob**: 192 cookies
- **Charlie**: 96 cookies
--------------------------------------------------------------------------------
Worker-worker:default/layer_0/worker_2:
To divide the cookies in the ratio of 3:4:2, we first need to find the total parts in the ratio. 

The total parts are:
- Alice: 3 parts
- Bob: 4 parts
- Charlie: 2 parts

Adding these parts together gives:
\[ 3 + 4 + 2 = 9 \text{ parts} \]

Next, we can determine how many cookies each part represents by dividing the total number of cookies by the total parts:
\[ \text{Cookies per part} = \frac{432 \text{ cookies}}{9 \text{ parts}} = 48 \text{ cookies/part} \]

Now we can calculate the number of cookies for each person:
- Alice's share: 
\[ 3 \text{ parts} \times 48 \text{ cookies/part} = 144 \text{ cookies} \]
- Bob's share: 
\[ 4 \text{ parts} \times 48 \text{ cookies/part} = 192 \text{ cookies} \]
- Charlie's share: 
\[ 2 \text{ parts} \times 48 \text{ cookies/part} = 96 \text{ cookies} \]

So, the final distribution of cookies is:
- Alice: 144 cookies
- Bob: 192 cookies
- Charlie: 96 cookies
--------------------------------------------------------------------------------
Orchestrator-orchestrator:default:
Received results from workers at layer 0
--------------------------------------------------------------------------------
Orchestrator-orchestrator:default:
Dispatch to workers at layer 1
--------------------------------------------------------------------------------
Worker-worker:default/layer_1/worker_2:
To divide 432 cookies in the ratio of 3:4:2 among Alice, Bob, and Charlie, follow these steps:

1. **Determine the total number of parts in the ratio**:
   \[
   3 + 4 + 2 = 9 \text{ parts}
   \]

2. **Calculate the value of one part** by dividing the total number of cookies by the total number of parts:
   \[
   \text{Value of one part} = \frac{432}{9} = 48
   \]

3. **Calculate the number of cookies each person receives** by multiplying the value of one part by the number of parts each individual gets:
   - **For Alice (3 parts)**:
     \[
     3 \times 48 = 144 \text{ cookies}
     \]
   - **For Bob (4 parts)**:
     \[
     4 \times 48 = 192 \text{ cookies}
     \]
   - **For Charlie (2 parts)**:
     \[
     2 \times 48 = 96 \text{ cookies}
     \]

Thus, the final distribution of cookies is:
- **Alice**: 144 cookies
- **Bob**: 192 cookies
- **Charlie**: 96 cookies
--------------------------------------------------------------------------------
Worker-worker:default/layer_1/worker_0:
To divide 432 cookies among Alice, Bob, and Charlie in the ratio of 3:4:2, we can follow these steps:

1. **Calculate the Total Parts**: 
   Add the parts of the ratio together:
   \[
   3 + 4 + 2 = 9 \text{ parts}
   \]

2. **Determine the Value of One Part**: 
   Divide the total number of cookies by the total number of parts:
   \[
   \text{Value of one part} = \frac{432 \text{ cookies}}{9 \text{ parts}} = 48 \text{ cookies/part}
   \]

3. **Calculate Each Person's Share**:
   - **Alice's Share** (3 parts):
     \[
     3 \times 48 = 144 \text{ cookies}
     \]
   - **Bob's Share** (4 parts):
     \[
     4 \times 48 = 192 \text{ cookies}
     \]
   - **Charlie's Share** (2 parts):
     \[
     2 \times 48 = 96 \text{ cookies}
     \]

4. **Final Distribution**:
   - Alice: 144 cookies
   - Bob: 192 cookies
   - Charlie: 96 cookies

Thus, the distribution of cookies is:
- **Alice**: 144 cookies
- **Bob**: 192 cookies
- **Charlie**: 96 cookies
--------------------------------------------------------------------------------
Worker-worker:default/layer_1/worker_1:
To divide 432 cookies among Alice, Bob, and Charlie in the ratio of 3:4:2, we first need to determine the total number of parts in this ratio.

1. **Calculate Total Parts:**
   \[
   3 \text{ (Alice)} + 4 \text{ (Bob)} + 2 \text{ (Charlie)} = 9 \text{ parts}
   \]

2. **Determine the Value of One Part:**
   Next, we'll find out how many cook
```


## Multi-Agent Debate

*[design-patterns/multi-agent-debate.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/multi-agent-debate.html)*

### Multi-Agent Debate

Multi-Agent Debate is a multi-agent design pattern that simulates a multi-turn interaction 
where in each turn, agents exchange their responses with each other, and refine 
their responses based on the responses from other agents.

This example shows an implementation of the multi-agent debate pattern for solving
math problems from the [GSM8K benchmark](https://huggingface.co/datasets/openai/gsm8k).

There are of two types of agents in this pattern: solver agents and an aggregator agent.
The solver agents are connected in a sparse manner following the technique described in
[Improving Multi-Agent Debate with Sparse Communication Topology](https://arxiv.org/abs/2406.11776).
The solver agents are responsible for solving math problems and exchanging responses with each other.
The aggregator agent is responsible for distributing math problems to the solver agents,
waiting for their final responses, and aggregating the responses to get the final answer.

The pattern works as follows:
1. User sends a math problem to the aggregator agent.
2. The aggregator agent distributes the problem to the solver agents.
3. Each solver agent processes the problem, and publishes a response to its neighbors.
4. Each solver agent uses the responses from its neighbors to refine its response, and publishes a new response.
5. Repeat step 4 for a fixed number of rounds. In the final round, each solver agent publishes a final response.
6. The aggregator agent uses majority voting to aggregate the final responses from all solver agents to get a final answer, and publishes the answer.

We will be using the broadcast API, i.e., {py:meth}`~autogen_core.BaseAgent.publish_message`,
and we will be using topic and subscription to implement the communication topology.
Read about [Topics and Subscriptions](../core-concepts/topic-and-subscription.md) to understand how they work.

```python
import re
from dataclasses import dataclass
from typing import Dict, List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

#### Message Protocol

First, we define the messages used by the agents.
`IntermediateSolverResponse` is the message exchanged among the solver agents in each round,
and `FinalSolverResponse` is the message published by the solver agents in the final round.

```python
@dataclass
class Question:
    content: str


@dataclass
class Answer:
    content: str


@dataclass
class SolverRequest:
    content: str
    question: str


@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int


@dataclass
class FinalSolverResponse:
    answer: str
```

#### Solver Agent

The solver agent is responsible for solving math problems and exchanging responses with other solver agents.
Upon receiving a `SolverRequest`, the solver agent uses an LLM to generate an answer.
Then, it publishes a `IntermediateSolverResponse`
or a `FinalSolverResponse` based on the round number.

The solver agent is given a topic type, which is used to indicate the topic
to which the agent should publish intermediate responses. This topic is subscribed
to by its neighbors to receive responses from this agent -- we will show
how this is done later.

We use {py:meth}`~autogen_core.components.default_subscription` to let
solver agents subscribe to the default topic, which is used by the aggregator agent
to collect the final responses from the solver agents.

```python
@default_subscription
class MathSolver(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, topic_type: str, num_neighbors: int, max_round: int) -> None:
        super().__init__("A debator.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateSolverResponse]] = {}
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant with expertise in mathematics and reasoning. "
                    "Your task is to assist in solving a math reasoning problem by providing "
                    "a clear and detailed solution. Limit your output within 100 words, "
                    "and your final answer should be a single numerical number, "
                    "in the form of {{answer}}, at the end of your response. "
                    "For example, 'The answer is {{42}}.'"
                )
            )
        ]
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx: MessageContext) -> None:
        # Add the question to the memory.
        self._history.append(UserMessage(content=message.content, source="user"))
        # Make an inference using the model.
        model_result = await self._model_client.create(self._system_messages + self._history)
        assert isinstance(model_result.content, str)
        # Add the response to the memory.
        self._history.append(AssistantMessage(content=model_result.content, source=self.metadata["type"]))
        print(f"{'-'*80}\nSolver {self.id} round {self._round}:\n{model_result.content}")
        # Extract the answer from the response.
        match = re.search(r"\{\{(\-?\d+(\.\d+)?)\}\}", model_result.content)
        if match is None:
            raise ValueError("The model response does not contain the answer.")
        answer = match.group(1)
        # Increment the counter.
        self._round += 1
        if self._round == self._max_round:
            # If the counter reaches the maximum round, publishes a final response.
            await self.publish_message(FinalSolverResponse(answer=answer), topic_id=DefaultTopicId())
        else:
            # Publish intermediate response to the topic associated with this solver.
            await self.publish_message(
                IntermediateSolverResponse(
                    content=model_result.content,
                    question=message.question,
                    answer=answer,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(type=self._topic_type),
            )

    @message_handler
    async def handle_response(self, message: IntermediateSolverResponse, ctx: MessageContext) -> None:
        # Add neighbor's response to the buffer.
        self._buffer.setdefault(message.round, []).append(message)
        # Check if all neighbors have responded.
        if len(self._buffer[message.round]) == self._num_neighbors:
            print(
                f"{'-'*80}\nSolver {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors."
            )
            # Prepare the prompt for the next question.
            prompt = "These are the solutions to the problem from other agents:\n"
            for resp in self._buffer[message.round]:
                prompt += f"One agent solution: {resp.content}\n"
            prompt += (
                "Using the solutions from other agents as additional information, "
                "can you provide your answer to the math problem? "
                f"The original math problem is {message.question}. "
                "Your final answer should be a single numerical number, "
                "in the form of {{answer}}, at the end of your response."
            )
            # Send the question to the agent itself to solve.
            await self.send_message(SolverRequest(content=prompt, question=message.question), self.id)
            # Clear the buffer.
            self._buffer.pop(message.round)
```

#### Aggregator Agent

The aggregator agent is responsible for handling user question and 
distributing math problems to the solver agents.

The aggregator subscribes to the default topic using
{py:meth}`~autogen_core.components.default_subscription`. The default topic is used to
recieve user question, receive the final responses from the solver agents,
and publish the final answer back to the user.

In a more complex application when you want to isolate the multi-agent debate into a
sub-component, you should use
{py:meth}`~autogen_core.components.type_subscription` to set a specific topic
type for the aggregator-solver communication, 
and have the both the solver and aggregator publish and subscribe to that topic type.

```python
@default_subscription
class MathAggregator(RoutedAgent):
    def __init__(self, num_solvers: int) -> None:
        super().__init__("Math Aggregator")
        self._num_solvers = num_solvers
        self._buffer: List[FinalSolverResponse] = []

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        print(f"{'-'*80}\nAggregator {self.id} received question:\n{message.content}")
        prompt = (
            f"Can you solve the following math problem?\n{message.content}\n"
            "Explain your reasoning. Your final answer should be a single numerical number, "
            "in the form of {{answer}}, at the end of your response."
        )
        print(f"{'-'*80}\nAggregator {self.id} publishes initial solver request.")
        await self.publish_message(SolverRequest(content=prompt, question=message.content), topic_id=DefaultTopicId())

    @message_handler
    async def handle_final_solver_response(self, message: FinalSolverResponse, ctx: MessageContext) -> None:
        self._buffer.append(message)
        if len(self._buffer) == self._num_solvers:
            print(f"{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            # Find the majority answer.
            answers = [resp.answer for resp in self._buffer]
            majority_answer = max(set(answers), key=answers.count)
            # Publish the aggregated response.
            await self.publish_message(Answer(content=majority_answer), topic_id=DefaultTopicId())
            # Clear the responses.
            self._buffer.clear()
            print(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}")
```

#### Setting Up a Debate

We will now set up a multi-agent debate with 4 solver agents and 1 aggregator agent.
The solver agents will be connected in a sparse manner as illustrated in the figure
below:

```
A --- B
|     |
|     |
D --- C
```

Each solver agent is connected to two other solver agents. 
For example, agent A is connected to agents B and C.

Let's first create a runtime and register the agent types.

```python
runtime = SingleThreadedAgentRuntime()

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

await MathSolver.register(
    runtime,
    "MathSolverA",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverA",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverB",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverB",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverC",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverC",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathSolver.register(
    runtime,
    "MathSolverD",
    lambda: MathSolver(
        model_client=model_client,
        topic_type="MathSolverD",
        num_neighbors=2,
        max_round=3,
    ),
)
await MathAggregator.register(runtime, "MathAggregator", lambda: MathAggregator(num_solvers=4))
```

```text
AgentType(type='MathAggregator')
```

Now we will create the solver agent topology using {py:class}`~autogen_core.components.TypeSubscription`,
which maps each solver agent's publishing topic type to its neighbors' agent types.

```python
# Subscriptions for topic published to by MathSolverA.
await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverD"))
await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverB"))

# Subscriptions for topic published to by MathSolverB.
await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverA"))
await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverC"))

# Subscriptions for topic published to by MathSolverC.
await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverB"))
await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverD"))

# Subscriptions for topic published to by MathSolverD.
await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverC"))
await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverA"))

# All solvers and the aggregator subscribe to the default topic.
```

#### Solving Math Problems

Now let's run the debate to solve a math problem.
We publish a `SolverRequest` to the default topic, 
and the aggregator agent will start the debate.

```python
question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
runtime.start()
await runtime.publish_message(Question(content=question), DefaultTopicId())
# Wait for the runtime to stop when idle.
await runtime.stop_when_idle()
# Close the connection to the model client.
await model_client.close()
```

```text
--------------------------------------------------------------------------------
Aggregator MathAggregator:default received question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
--------------------------------------------------------------------------------
Aggregator MathAggregator:default publishes initial solver request.
--------------------------------------------------------------------------------
Solver MathSolverC:default round 0:
In April, Natalia sold 48 clips. In May, she sold half as many, which is 48 / 2 = 24 clips. To find the total number of clips sold in April and May, we add the amounts: 48 (April) + 24 (May) = 72 clips. 

Thus, the total number of clips sold by Natalia is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverB:default round 0:
In April, Natalia sold 48 clips. In May, she sold half as many clips, which is 48 / 2 = 24 clips. To find the total clips sold in April and May, we add both amounts: 

48 (April) + 24 (May) = 72.

Thus, the total number of clips sold altogether is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverD:default round 0:
Natalia sold 48 clips in April. In May, she sold half as many, which is \( \frac{48}{2} = 24 \) clips. To find the total clips sold in both months, we add the clips sold in April and May together:

\[ 48 + 24 = 72 \]

Thus, Natalia sold a total of 72 clips.

The answer is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverC:default round 1:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverA:default round 1:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverA:default round 0:
In April, Natalia sold clips to 48 friends. In May, she sold half as many, which is calculated as follows:

Half of 48 is \( 48 \div 2 = 24 \).

Now, to find the total clips sold in April and May, we add the totals from both months:

\( 48 + 24 = 72 \).

Thus, the total number of clips Natalia sold altogether in April and May is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverD:default round 1:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverB:default round 1:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverC:default round 1:
In April, Natalia sold 48 clips. In May, she sold half as many, which is 48 / 2 = 24 clips. The total number of clips sold in April and May is calculated by adding the two amounts: 48 (April) + 24 (May) = 72 clips. 

Therefore, the answer is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverA:default round 1:
In April, Natalia sold 48 clips. In May, she sold half of that amount, which is 48 / 2 = 24 clips. To find the total clips sold in both months, we sum the clips from April and May: 

48 (April) + 24 (May) = 72.

Thus, Natalia sold a total of {{72}} clips. 

The answer is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverD:default round 2:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverB:default round 2:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverD:default round 1:
Natalia sold 48 clips in April. In May, she sold half of that, which is \( 48 \div 2 = 24 \) clips. To find the total clips sold, we add the clips sold in both months:

\[ 48 + 24 = 72 \]

Therefore, the total number of clips sold by Natalia is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverB:default round 1:
In April, Natalia sold 48 clips. In May, she sold half that amount, which is 48 / 2 = 24 clips. To find the total clips sold in both months, we add the amounts: 

48 (April) + 24 (May) = 72.

Therefore, the total number of clips sold altogether by Natalia is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverA:default round 2:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverC:default round 2:
Received all responses from 2 neighbors.
--------------------------------------------------------------------------------
Solver MathSolverA:default round 2:
In April, Natalia sold 48 clips. In May, she sold half of that amount, which is \( 48 \div 2 = 24 \) clips. To find the total clips sold in both months, we add the amounts from April and May:

\( 48 + 24 = 72 \).

Thus, the total number of clips sold by Natalia is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverC:default round 2:
In April, Natalia sold 48 clips. In May, she sold half of that amount, which is \( 48 \div 2 = 24 \) clips. To find the total number of clips sold in both months, we add the clips sold in April and May: 

48 (April) + 24 (May) = 72. 

Thus, the total number of clips sold altogether by Natalia is {{72}}.
--------------------------------------------------------------------------------
Solver MathSolverB:default round 2:
In April, Natalia sold 48 clips. In May, she sold half as many, calculated as \( 48 \div 2 = 24 \) clips. To find the total clips sold over both months, we sum the totals: 

\( 48 (April) + 24 (May) = 72 \).

Therefore, the total number of cl
```


## Reflection

*[design-patterns/reflection.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html)*

### Reflection

Reflection is a design pattern where an LLM generation is followed by a reflection,
which in itself is another LLM generation conditioned on the output of the first one.
For example, given a task to write code, the first LLM can generate a code snippet,
and the second LLM can generate a critique of the code snippet.

In the context of AutoGen and agents, reflection can be implemented as a pair
of agents, where the first agent generates a message and the second agent
generates a response to the message. The two agents continue to interact
until they reach a stopping condition, such as a maximum number of iterations
or an approval from the second agent.

Let's implement a simple reflection design pattern using AutoGen agents.
There will be two agents: a coder agent and a reviewer agent, the coder agent
will generate a code snippet, and the reviewer agent will generate a critique
of the code snippet.

#### Message Protocol

Before we define the agents, we need to first define the message protocol for the agents.

```python
from dataclasses import dataclass


@dataclass
class CodeWritingTask:
    task: str


@dataclass
class CodeWritingResult:
    task: str
    code: str
    review: str


@dataclass
class CodeReviewTask:
    session_id: str
    code_writing_task: str
    code_writing_scratchpad: str
    code: str


@dataclass
class CodeReviewResult:
    review: str
    session_id: str
    approved: bool
```

The above set of messages defines the protocol for our example reflection design pattern:
- The application sends a `CodeWritingTask` message to the coder agent
- The coder agent generates a `CodeReviewTask` message, which is sent to the reviewer agent
- The reviewer agent generates a `CodeReviewResult` message, which is sent back to the coder agent
- Depending on the `CodeReviewResult` message, if the code is approved, the coder agent sends a `CodeWritingResult` message
back to the application, otherwise, the coder agent sends another `CodeReviewTask` message to the reviewer agent,
and the process continues.

We can visualize the message protocol using a data flow diagram:

![coder-reviewer data flow](coder-reviewer-data-flow.svg)

#### Agents

Now, let's define the agents for the reflection design pattern.

```python
import json
import re
import uuid
from typing import Dict, List, Union

from autogen_core import MessageContext, RoutedAgent, TopicId, default_subscription, message_handler
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
```

We use the [Broadcast](../framework/message-and-communication.ipynb#broadcast) API
to implement the design pattern. The agents implements the pub/sub model.
The coder agent subscribes to the `CodeWritingTask` and `CodeReviewResult` messages,
and publishes the `CodeReviewTask` and `CodeWritingResult` messages.

````python
@default_subscription
class CoderAgent(RoutedAgent):
    """An agent that performs code writing tasks."""

    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A code writing agent.")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a proficient coder. You write code to solve problems.
Work with the reviewer to improve your code.
Always put all finished code in a single Markdown code block.
For example:
```python
def hello_world():
    print("Hello, World!")
```

Respond using the following format:

Thoughts: <Your comments>
Code: <Your code>
""",
            )
        ]
        self._model_client = model_client
        self._session_memory: Dict[str, List[CodeWritingTask | CodeReviewTask | CodeReviewResult]] = {}

    @message_handler
    async def handle_code_writing_task(self, message: CodeWritingTask, ctx: MessageContext) -> None:
        # Store the messages in a temporary memory for this request only.
        session_id = str(uuid.uuid4())
        self._session_memory.setdefault(session_id, []).append(message)
        # Generate a response using the chat completion API.
        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=message.task, source=self.metadata["type"])],
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        # Extract the code block from the response.
        code_block = self._extract_code_block(response.content)
        if code_block is None:
            raise ValueError("Code block not found.")
        # Create a code review task.
        code_review_task = CodeReviewTask(
            session_id=session_id,
            code_writing_task=message.task,
            code_writing_scratchpad=response.content,
            code=code_block,
        )
        # Store the code review task in the session memory.
        self._session_memory[session_id].append(code_review_task)
        # Publish a code review task.
        await self.publish_message(code_review_task, topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_code_review_result(self, message: CodeReviewResult, ctx: MessageContext) -> None:
        # Store the review result in the session memory.
        self._session_memory[message.session_id].append(message)
        # Obtain the request from previous messages.
        review_request = next(
            m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, CodeReviewTask)
        )
        assert review_request is not None
        # Check if the code is approved.
        if message.approved:
            # Publish the code writing result.
            await self.publish_message(
                CodeWritingResult(
                    code=review_request.code,
                    task=review_request.code_writing_task,
                    review=message.review,
                ),
                topic_id=TopicId("default", self.id.key),
            )
            print("Code Writing Result:")
            print("-" * 80)
            print(f"Task:\n{review_request.code_writing_task}")
            print("-" * 80)
            print(f"Code:\n{review_request.code}")
            print("-" * 80)
            print(f"Review:\n{message.review}")
            print("-" * 80)
        else:
            # Create a list of LLM messages to send to the model.
            messages: List[LLMMessage] = [*self._system_messages]
            for m in self._session_memory[message.session_id]:
                if isinstance(m, CodeReviewResult):
                    messages.append(UserMessage(content=m.review, source="Reviewer"))
                elif isinstance(m, CodeReviewTask):
                    messages.append(AssistantMessage(content=m.code_writing_scratchpad, source="Coder"))
                elif isinstance(m, CodeWritingTask):
                    messages.append(UserMessage(content=m.task, source="User"))
                else:
                    raise ValueError(f"Unexpected message type: {m}")
            # Generate a revision using the chat completion API.
            response = await self._model_client.create(messages, cancellation_token=ctx.cancellation_token)
            assert isinstance(response.content, str)
            # Extract the code block from the response.
            code_block = self._extract_code_block(response.content)
            if code_block is None:
                raise ValueError("Code block not found.")
            # Create a new code review task.
            code_review_task = CodeReviewTask(
                session_id=message.session_id,
                code_writing_task=review_request.code_writing_task,
                code_writing_scratchpad=response.content,
                code=code_block,
            )
            # Store the code review task in the session memory.
            self._session_memory[message.session_id].append(code_review_task)
            # Publish a new code review task.
            await self.publish_message(code_review_task, topic_id=TopicId("default", self.id.key))

    def _extract_code_block(self, markdown_text: str) -> Union[str, None]:
        pattern = r"```(\w+)\n(.*?)\n```"
        # Search for the pattern in the markdown text
        match = re.search(pattern, markdown_text, re.DOTALL)
        # Extract the language and code block if a match is found
        if match:
            return match.group(2)
        return None
````

A few things to note about `CoderAgent`:
- It uses chain-of-thought prompting in its system message.
- It stores message histories for different `CodeWritingTask` in a dictionary,
so each task has its own history.
- When making an LLM inference request using its model client, it transforms
the message history into a list of {py:class}`autogen_core.models.LLMMessage` objects
to pass to the model client.

The reviewer agent subscribes to the `CodeReviewTask` message and publishes the `CodeReviewResult` message.

````python
@default_subscription
class ReviewerAgent(RoutedAgent):
    """An agent that performs code review tasks."""

    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A code reviewer agent.")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a code reviewer. You focus on correctness, efficiency and safety of the code.
Respond using the following JSON format:
{
    "correctness": "<Your comments>",
    "efficiency": "<Your comments>",
    "safety": "<Your comments>",
    "approval": "<APPROVE or REVISE>",
    "suggested_changes": "<Your comments>"
}
""",
            )
        ]
        self._session_memory: Dict[str, List[CodeReviewTask | CodeReviewResult]] = {}
        self._model_client = model_client

    @message_handler
    async def handle_code_review_task(self, message: CodeReviewTask, ctx: MessageContext) -> None:
        # Format the prompt for the code review.
        # Gather the previous feedback if available.
        previous_feedback = ""
        if message.session_id in self._session_memory:
            previous_review = next(
                (m for m in reversed(self._session_memory[message.session_id]) if isinstance(m, CodeReviewResult)),
                None,
            )
            if previous_review is not None:
                previous_feedback = previous_review.review
        # Store the messages in a temporary memory for this request only.
        self._session_memory.setdefault(message.session_id, []).append(message)
        prompt = f"""The problem statement is: {message.code_writing_task}
The code is:
```
{message.code}
```

Previous feedback:
{previous_feedback}

Please review the code. If previous feedback was provided, see if it was addressed.
"""
        # Generate a response using the chat completion API.
        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=prompt, source=self.metadata["type"])],
            cancellation_token=ctx.cancellation_token,
            json_output=True,
        )
        assert isinstance(response.content, str)
        # TODO: use structured generation library e.g. guidance to ensure the response is in the expected format.
        # Parse the response JSON.
        review = json.loads(response.content)
        # Construct the review text.
        review_text = "Code review:\n" + "\n".join([f"{k}: {v}" for k, v in review.items()])
        approved = review["approval"].lower().strip() == "approve"
        result = CodeReviewResult(
            review=review_text,
            session_id=message.session_id,
            approved=approved,
        )
        # Store the review result in the session memory.
        self._session_memory[message.session_id].append(result)
        # Publish the review result.
        await self.publish_message(result, topic_id=TopicId("default", self.id.key))
````

The `ReviewerAgent` uses JSON-mode when making an LLM inference request, and
also uses chain-of-thought prompting in its system message.

#### Logging

Turn on logging to see the messages exchanged between the agents.

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
```

#### Running the Design Pattern

Let's test the design pattern with a coding task.
Since all the agents are decorated with the {py:meth}`~autogen_core.components.default_subscription` class decorator,
the agents when created will automatically subscribe to the default topic.
We publish a `CodeWritingTask` message to the default topic to start the reflection process.

```python
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient

runtime = SingleThreadedAgentRuntime()
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
await ReviewerAgent.register(runtime, "ReviewerAgent", lambda: ReviewerAgent(model_client=model_client))
await CoderAgent.register(runtime, "CoderAgent", lambda: CoderAgent(model_client=model_client))
runtime.start()
await runtime.publish_message(
    message=CodeWritingTask(task="Write a function to find the sum of all even numbers in a list."),
    topic_id=DefaultTopicId(),
)

# Keep processing messages until idle.
await runtime.stop_when_idle()
# Close the model client.
await model_client.close()
```

````text
INFO:autogen_core:Publishing message of type CodeWritingTask to all subscribers: {'task': 'Write a function to find the sum of all even numbers in a list.'}
INFO:autogen_core:Calling message handler for ReviewerAgent with message type CodeWritingTask published by Unknown
INFO:autogen_core:Calling message handler for CoderAgent with message type CodeWritingTask published by Unknown
INFO:autogen_core:Unhandled message: CodeWritingTask(task='Write a function to find the sum of all even numbers in a list.')
INFO:autogen_core.events:{"prompt_tokens": 101, "completion_tokens": 88, "type": "LLMCall"}
INFO:autogen_core:Publishing message of type CodeReviewTask to all subscribers: {'session_id': '51db93d5-3e29-4b7f-9f96-77be7bb02a5e', 'code_writing_task': 'Write a function to find the sum of all even numbers in a list.', 'code_writing_scratchpad': 'Thoughts: To find the sum of all even numbers in a list, we can use a list comprehension to filter out the even numbers and then use the `sum()` function to calculate their total. The implementation should handle edge cases like an empty list or a list with no even numbers.\n\nCode:\n```python\ndef sum_of_even_numbers(numbers):\n    return sum(num for num in numbers if num % 2 == 0)\n```', 'code': 'def sum_of_even_numbers(numbers):\n    return sum(num for num in numbers if num % 2 == 0)'}
INFO:autogen_core:Calling message handler for ReviewerAgent with message type CodeReviewTask published by CoderAgent:default
INFO:autogen_core.events:{"prompt_tokens": 163, "completion_tokens": 235, "type": "LLMCall"}
INFO:autogen_core:Publishing message of type CodeReviewResult to all subscribers: {'review': "Code review:\ncorrectness: The function correctly identifies and sums all even numbers in the provided list. The use of a generator expression ensures that only even numbers are processed, which is correct.\nefficiency: The function is efficient as it utilizes a generator expression that avoids creating an intermediate list, therefore using less memory. The time complexity is O(n) where n is the number of elements in the input list, which is optimal for this task.\nsafety: The function does not include checks for input types. If a non-iterable or a list containing non-integer types is passed, it could lead to unexpected behavior or errors. It’s advisable to handle such cases.\napproval: REVISE\nsuggested_changes: Consider adding input validation to ensure that 'numbers' is a list and contains only integers. You could raise a ValueError if the input is invalid. Example: 'if not isinstance(numbers, list) or not all(isinstance(num, int) for num in numbers): raise ValueError('Input must be a list of integers')'. This will make the function more robust.", 'session_id': '51db93d5-3e29-4b7f-9f96-77be7bb02a5e', 'approved': False}
INFO:autogen_core:Calling message handler for CoderAgent with message type CodeReviewResult published by ReviewerAgent:default
INFO:autogen_core.events:{"prompt_tokens": 421, "completion_tokens": 119, "type": "LLMCall"}
INFO:autogen_core:Publishing message of type CodeReviewTask to all subscribers: {'session_id': '51db93d5-3e29-4b7f-9f96-77be7bb02a5e', 'code_writing_task': 'Write a function to find the sum of all even numbers in a list.', 'code_writing_scratchpad': "Thoughts: I appreciate the reviewer's feedback on input validation. Adding type checks ensures that the function can handle unexpected inputs gracefully. I will implement the suggested changes and include checks for both the input type and the elements within the list to confirm that they are integers.\n\nCode:\n```python\ndef sum_of_even_numbers(numbers):\n    if not isinstance(numbers, list) or not all(isinstance(num, int) for num in numbers):\n        raise ValueError('Input must be a list of integers')\n    \n    return sum(num for num in numbers if num % 2 == 0)\n```", 'code': "def sum_of_even_numbers(numbers):\n    if not isinstance(numbers, list) or not all(isinstance(num, int) for num in numbers):\n        raise ValueError('Input must be a list of integers')\n    \n    return sum(num for num in numbers if num % 2 == 0)"}
INFO:autogen_core:Calling message handler for ReviewerAgent with message type CodeReviewTask published by CoderAgent:default
INFO:autogen_core.events:{"prompt_tokens": 420, "completion_tokens": 153, "type": "LLMCall"}
INFO:autogen_core:Publishing message of type CodeReviewResult to all subscribers: {'review': 'Code review:\ncorrectness: The function correctly sums all even numbers in the provided list. It raises a ValueError if the input is not a list of integers, which is a necessary check for correctness.\nefficiency: The function remains efficient with a time complexity of O(n) due to the use of a generator expression. There are no unnecessary intermediate lists created, so memory usage is optimal.\nsafety: The function includes input validation, which enhances safety by preventing incorrect input types. It raises a ValueError for invalid inputs, making the function more robust against unexpected data.\napproval: APPROVE\nsuggested_changes: No further changes are necessary as the previous feedback has been adequately addressed.', 'session_id': '51db93d5-3e29-4b7f-9f96-77be7bb02a5e', 'approved': True}
INFO:autogen_core:Calling message handler for CoderAgent with message type CodeReviewResult published by ReviewerAgent:default
INFO:autogen_core:Publishing message of type CodeWritingResult to all subscribers: {'task': 'Write a function to find the sum of all even numbers in a list.', 'code': "def sum_of_even_numbers(numbers):\n    if not isinstance(numbers, list) or not all(isinstance(num, int) for num in numbers):\n        raise ValueError('Input must be a list of integers')\n    \n    return sum(num for num in numbers if num % 2 == 0)", 'review': 'Code review:\ncorrectness: The function correctly sums all even numbers in the provided list. It raises a ValueError if the input is not a list of integers, which is a necessary check for correctness.\nefficiency
````

The log messages show the interaction between the coder and reviewer agents.
The final output shows the code snippet generated by the coder agent and the critique generated by the reviewer agent.


## Code Execution

*[design-patterns/code-execution-groupchat.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/code-execution-groupchat.html)*

### Code Execution

In this section we explore creating custom agents to handle code generation and execution. These tasks can be handled using the provided Agent implementations found here {py:meth}`~autogen_agentchat.agents.AssistantAgent`, {py:meth}`~autogen_agentchat.agents.CodeExecutorAgent`; but this guide will show you how to implement custom, lightweight agents that can replace their functionality. This simple example implements two agents that create a plot of Tesla's and Nvidia's stock returns.

We first define the agent classes and their respective procedures for 
handling messages.
We create two agent classes: `Assistant` and `Executor`. The `Assistant`
agent writes code and the `Executor` agent executes the code.
We also create a `Message` data class, which defines the messages that are passed between
the agents.

```{attention}
Code generated in this example is run within a [Docker](https://www.docker.com/) container. Please ensure Docker is [installed](https://docs.docker.com/get-started/get-docker/) and running prior to running the example. Local code execution is available ({py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`) but is not recommended due to the risk of running LLM generated code in your local environment.
```

````python
import re
from dataclasses import dataclass
from typing import List

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)


@dataclass
class Message:
    content: str


@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""Write Python script in markdown block, and it will be executed.
Always save figures to file in the current directory. Do not use plt.show(). All code required to complete this task must be contained within a single response.""",
            )
        ]

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}")
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
        await self.publish_message(Message(content=result.content), DefaultTopicId())  # type: ignore


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        code_blocks = extract_markdown_code_blocks(message.content)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            print(f"\n{'-'*80}\nExecutor:\n{result.output}")
            await self.publish_message(Message(content=result.output), DefaultTopicId())
````

You might have already noticed, the agents' logic, whether it is using model or code executor,
is completely decoupled from
how messages are delivered. This is the core idea: the framework provides
a communication infrastructure, and the agents are responsible for their own
logic. We call the communication infrastructure an **Agent Runtime**.

Agent runtime is a key concept of this framework. Besides delivering messages,
it also manages agents' lifecycle. 
So the creation of agents are handled by the runtime.

The following code shows how to register and run the agents using 
{py:class}`~autogen_core.SingleThreadedAgentRuntime`,
a local embedded agent runtime implementation.

```python
import tempfile

from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

work_dir = tempfile.mkdtemp()

# Create an local embedded runtime.
runtime = SingleThreadedAgentRuntime()

async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:  # type: ignore[syntax]
    # Register the assistant and executor agents by providing
    # their agent types, the factory functions for creating instance and subscriptions.
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        # api_key="YOUR_API_KEY"
    )
    await Assistant.register(
        runtime,
        "assistant",
        lambda: Assistant(model_client=model_client),
    )
    await Executor.register(runtime, "executor", lambda: Executor(executor))

    # Start the runtime and publish a message to the assistant.
    runtime.start()
    await runtime.publish_message(
        Message("Create a plot of NVIDA vs TSLA stock returns YTD from 2024-01-01."), DefaultTopicId()
    )

    # Wait for the runtime to stop when idle.
    await runtime.stop_when_idle()
    # Close the connection to the model client.
    await model_client.close()
```

````text
--------------------------------------------------------------------------------
Assistant:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the ticker symbols for NVIDIA and Tesla
tickers = ['NVDA', 'TSLA']

# Download the stock data from Yahoo Finance starting from 2024-01-01
start_date = '2024-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

# Download the adjusted closing prices
stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate the daily returns
returns = stock_data.pct_change().dropna()

# Plot the cumulative returns for each stock
cumulative_returns = (1 + returns).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns['NVDA'], label='NVIDIA', color='green')
plt.plot(cumulative_returns.index, cumulative_returns['TSLA'], label='Tesla', color='red')
plt.title('NVIDIA vs Tesla Stock Returns YTD (2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig('nvidia_vs_tesla_ytd_returns.png')
```

--------------------------------------------------------------------------------
Executor:
Traceback (most recent call last):
  File "/workspace/tmp_code_fd7395dcad4fbb74d40c981411db604e78e1a17783ca1fab3aaec34ff2c3fdf0.python", line 1, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'


--------------------------------------------------------------------------------
Assistant:
It seems like the necessary libraries are not available in your environment. However, since I can't install packages or check the environment directly from here, you'll need to make sure that the appropriate packages are installed in your working environment. Once the modules are available, the script provided will execute properly.

Here's how you can install the required packages using pip (make sure to run these commands in your terminal or command prompt):

```bash
pip install pandas matplotlib yfinance
```

Let me provide you the script again for reference:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the ticker symbols for NVIDIA and Tesla
tickers = ['NVDA', 'TSLA']

# Download the stock data from Yahoo Finance starting from 2024-01-01
start_date = '2024-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

# Download the adjusted closing prices
stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate the daily returns
returns = stock_data.pct_change().dropna()

# Plot the cumulative returns for each stock
cumulative_returns = (1 + returns).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns['NVDA'], label='NVIDIA', color='green')
plt.plot(cumulative_returns.index, cumulative_returns['TSLA'], label='Tesla', color='red')
plt.title('NVIDIA vs Tesla Stock Returns YTD (2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig('nvidia_vs_tesla_ytd_returns.png')
```

Make sure to install the packages in the environment where you run this script. Feel free to ask if you have further questions or issues!

--------------------------------------------------------------------------------
Executor:
[*********************100%***********************]  2 of 2 completed


--------------------------------------------------------------------------------
Assistant:
It looks like the data fetching process completed successfully. You should now have a plot saved as `nvidia_vs_tesla_ytd_returns.png` in your current directory. If you have any additional questions or need further assistance, feel free to ask!
````

From the agent's output, we can see the plot of Tesla's and Nvidia's stock returns
has been created.

```python
from IPython.display import Image

Image(filename=f"{work_dir}/nvidia_vs_tesla_ytd_returns.png")  # type: ignore
```

```text
<IPython.core.display.Image object>
```

AutoGen also supports a distributed agent runtime, which can host agents running on
different processes or machines, with different identities, languages and dependencies.

To learn how to use agent runtime, communication, message handling, and subscription, please continue
reading the sections following this quick start.


---

# Cookbook


## Cookbook

*[cookbook/index.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/index.html)*

### Cookbook

This section contains a collection of recipes that demonstrate how to use the Core API features.

#### List of recipes

```{toctree}
:maxdepth: 1

azure-openai-with-aad-auth
termination-with-intervention
tool-use-with-intervention
extracting-results-with-an-agent
openai-assistant-agent
langgraph-agent
llamaindex-agent
local-llms-ollama-litellm
instrumenting
topic-subscription-scenarios
structured-output-agent
llm-usage-logger
```



## Azure OpenAI with AAD Auth

*[cookbook/azure-openai-with-aad-auth.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/azure-openai-with-aad-auth.html)*

### Azure OpenAI with AAD Auth

This guide will show you how to use the Azure OpenAI client with Azure Active Directory (AAD) authentication.

The identity used must be assigned the [**Cognitive Services OpenAI User**](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/role-based-access-control#cognitive-services-openai-user) role.

#### Install Azure Identity client

The Azure identity client is used to authenticate with Azure Active Directory.

```sh
pip install azure-identity
```

#### Using the Model Client

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Create the token provider
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAIChatCompletionClient(
    azure_deployment="{your-azure-deployment}",
    model="{model-name, such as gpt-4o}",
    api_version="2024-02-01",
    azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
    azure_ad_token_provider=token_provider,
)
```

```{note}
See [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity#chat-completions) for how to use the Azure client directly or for more info.
```



## Termination using Intervention Handler

*[cookbook/termination-with-intervention.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/termination-with-intervention.html)*

### Termination using Intervention Handler

```{note}
This method is valid when using {py:class}`~autogen_core.SingleThreadedAgentRuntime`.
```

There are many different ways to handle termination in `autogen_core`. Ultimately, the goal is to detect that the runtime no longer needs to be executed and you can proceed to finalization tasks. One way to do this is to use an {py:class}`autogen_core.base.intervention.InterventionHandler` to detect a termination message and then act on it.

```python
from dataclasses import dataclass
from typing import Any

from autogen_core import (
    DefaultInterventionHandler,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
```

First, we define a dataclass for regular message and message that will be used to signal termination.

```python
@dataclass
class Message:
    content: Any


@dataclass
class Termination:
    reason: str
```

We code our agent to publish a termination message when it decides it is time to terminate.

```python
@default_subscription
class AnAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")
        self.received = 0

    @message_handler
    async def on_new_message(self, message: Message, ctx: MessageContext) -> None:
        self.received += 1
        if self.received > 3:
            await self.publish_message(Termination(reason="Reached maximum number of messages"), DefaultTopicId())
```

Next, we create an InterventionHandler that will detect the termination message and act on it. This one hooks into publishes and when it encounters `Termination` it alters its internal state to indicate that termination has been requested.

```python
class TerminationHandler(DefaultInterventionHandler):
    def __init__(self) -> None:
        self._termination_value: Termination | None = None

    async def on_publish(self, message: Any, *, message_context: MessageContext) -> Any:
        if isinstance(message, Termination):
            self._termination_value = message
        return message

    @property
    def termination_value(self) -> Termination | None:
        return self._termination_value

    @property
    def has_terminated(self) -> bool:
        return self._termination_value is not None
```

Finally, we add this handler to the runtime and use it to detect termination and stop the runtime when the termination message is received.

```python
termination_handler = TerminationHandler()
runtime = SingleThreadedAgentRuntime(intervention_handlers=[termination_handler])

await AnAgent.register(runtime, "my_agent", AnAgent)

runtime.start()

# Publish more than 3 messages to trigger termination.
await runtime.publish_message(Message("hello"), DefaultTopicId())
await runtime.publish_message(Message("hello"), DefaultTopicId())
await runtime.publish_message(Message("hello"), DefaultTopicId())
await runtime.publish_message(Message("hello"), DefaultTopicId())

# Wait for termination.
await runtime.stop_when(lambda: termination_handler.has_terminated)

print(termination_handler.termination_value)
```

```text
Termination(reason='Reached maximum number of messages')
```


## User Approval for Tool Execution using Intervention Handler

*[cookbook/tool-use-with-intervention.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/tool-use-with-intervention.html)*

### User Approval for Tool Execution using Intervention Handler

This cookbook shows how to intercept the tool execution using
an intervention hanlder, and prompt the user for permission to execute the tool.

```python
from dataclasses import dataclass
from typing import Any, List

from autogen_core import (
    AgentId,
    AgentType,
    DefaultInterventionHandler,
    DropMessage,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tool_agent import ToolAgent, ToolException, tool_agent_caller_loop
from autogen_core.tools import ToolSchema
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
```

Let's define a simple message type that carries a string content.

```python
@dataclass
class Message:
    content: str
```

Let's create a simple tool use agent that is capable of using tools through a
{py:class}`~autogen_core.tool_agent.ToolAgent`.

```python
class ToolUseAgent(RoutedAgent):
    """An agent that uses tools to perform tasks. It executes the tools
    by itself by sending the tool execution task to a ToolAgent."""

    def __init__(
        self,
        description: str,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        tool_schema: List[ToolSchema],
        tool_agent_type: AgentType,
    ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._system_messages = system_messages
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(type=tool_agent_type, key=self.id.key)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        """Handle a user message, execute the model and tools, and returns the response."""
        session: List[LLMMessage] = [UserMessage(content=message.content, source="User")]
        # Use the tool agent to execute the tools, and get the output messages.
        output_messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        # Extract the final response from the output messages.
        final_response = output_messages[-1].content
        assert isinstance(final_response, str)
        return Message(content=final_response)
```

The tool use agent sends tool call requests to the tool agent to execute tools,
so we can intercept the messages sent by the tool use agent to the tool agent
to prompt the user for permission to execute the tool.

Let's create an intervention handler that intercepts the messages and prompts
user for before allowing the tool execution.

```python
class ToolInterventionHandler(DefaultInterventionHandler):
    async def on_send(
        self, message: Any, *, message_context: MessageContext, recipient: AgentId
    ) -> Any | type[DropMessage]:
        if isinstance(message, FunctionCall):
            # Request user prompt for tool execution.
            user_input = input(
                f"Function call: {message.name}\nArguments: {message.arguments}\nDo you want to execute the tool? (y/n): "
            )
            if user_input.strip().lower() != "y":
                raise ToolException(content="User denied tool execution.", call_id=message.id, name=message.name)
        return message
```

Now, we can create a runtime with the intervention handler registered.

```python
# Create the runtime with the intervention handler.
runtime = SingleThreadedAgentRuntime(intervention_handlers=[ToolInterventionHandler()])
```

In this example, we will use a tool for Python code execution.
First, we create a Docker-based command-line code executor
using {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`,
and then use it to instantiate a built-in Python code execution tool
{py:class}`~autogen_core.tools.PythonCodeExecutionTool`
that runs code in a Docker container.

```python
# Create the docker executor for the Python code execution tool.
docker_executor = DockerCommandLineCodeExecutor()

# Create the Python code execution tool.
python_tool = PythonCodeExecutionTool(executor=docker_executor)
```

Register the agents with tools and tool schema.

```python
# Register agents.
tool_agent_type = await ToolAgent.register(
    runtime,
    "tool_executor_agent",
    lambda: ToolAgent(
        description="Tool Executor Agent",
        tools=[python_tool],
    ),
)
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
await ToolUseAgent.register(
    runtime,
    "tool_enabled_agent",
    lambda: ToolUseAgent(
        description="Tool Use Agent",
        system_messages=[SystemMessage(content="You are a helpful AI Assistant. Use your tools to solve problems.")],
        model_client=model_client,
        tool_schema=[python_tool.schema],
        tool_agent_type=tool_agent_type,
    ),
)
```

```text
AgentType(type='tool_enabled_agent')
```

Run the agents by starting the runtime and sending a message to the tool use agent.
The intervention handler will prompt you for permission to execute the tool.

```python
# Start the runtime and the docker executor.
await docker_executor.start()
runtime.start()

# Send a task to the tool user.
response = await runtime.send_message(
    Message("Run the following Python code: print('Hello, World!')"), AgentId("tool_enabled_agent", "default")
)
print(response.content)

# Stop the runtime and the docker executor.
await runtime.stop()
await docker_executor.stop()

# Close the connection to the model client.
await model_client.close()
```

```text
The output of the code is: **Hello, World!**
```


## Extracting Results with an Agent

*[cookbook/extracting-results-with-an-agent.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/extracting-results-with-an-agent.html)*

### Extracting Results with an Agent

When running a multi-agent system to solve some task, you may want to extract the result of the system once it has reached termination. This guide showcases one way to achieve this. Given that agent instances are not directly accessible from the outside, we will use an agent to publish the final result to an accessible location.

If you model your system to publish some `FinalResult` type then you can create an agent whose sole job is to subscribe to this and make it available externally. For simple agents like this the {py:class}`~autogen_core.components.ClosureAgent` is an option to reduce the amount of boilerplate code. This allows you to define a function that will be associated as the agent's message handler. In this example, we're going to use a queue shared between the agent and the external code to pass the result.

```{note}
When considering how to extract results from a multi-agent system, you must always consider the subscriptions of the agent and the topics they publish to.
This is because the agent will only receive messages from topics it is subscribed to.
```

```python
import asyncio
from dataclasses import dataclass

from autogen_core import (
    ClosureAgent,
    ClosureContext,
    DefaultSubscription,
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
)
```

Define a dataclass for the final result.

```python
@dataclass
class FinalResult:
    value: str
```

Create a queue to pass the result from the agent to the external code.

```python
queue = asyncio.Queue[FinalResult]()
```

Create a function closure for outputting the final result to the queue.
The function must follow the signature
`Callable[[AgentRuntime, AgentId, T, MessageContext], Awaitable[Any]]`
where `T` is the type of the message the agent will receive.
You can use union types to handle multiple message types.

```python
async def output_result(_agent: ClosureContext, message: FinalResult, ctx: MessageContext) -> None:
    await queue.put(message)
```

Let's create a runtime and register a {py:class}`~autogen_core.components.ClosureAgent` that will publish the final result to the queue.

```python
runtime = SingleThreadedAgentRuntime()
await ClosureAgent.register_closure(
    runtime, "output_result", output_result, subscriptions=lambda: [DefaultSubscription()]
)
```

```text
AgentType(type='output_result')
```

We can simulate the collection of final results by publishing them directly to the runtime.

```python
runtime.start()
await runtime.publish_message(FinalResult("Result 1"), DefaultTopicId())
await runtime.publish_message(FinalResult("Result 2"), DefaultTopicId())
await runtime.stop_when_idle()
```

We can take a look at the queue to see the final result.

```python
while not queue.empty():
    print((result := await queue.get()).value)
```

```text
Result 1
Result 2
```


## OpenAI Assistant Agent

*[cookbook/openai-assistant-agent.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/openai-assistant-agent.html)*

### OpenAI Assistant Agent

[Open AI Assistant](https://platform.openai.com/docs/assistants/overview) 
and [Azure OpenAI Assistant](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant)
are server-side APIs for building
agents.
They can be used to build agents in AutoGen. This cookbook demonstrates how to
to use OpenAI Assistant to create an agent that can run code and Q&A over document.

#### Message Protocol

First, we need to specify the message protocol for the agent backed by 
OpenAI Assistant. The message protocol defines the structure of messages
handled and published by the agent. 
For illustration, we define a simple
message protocol of 4 message types: `Message`, `Reset`, `UploadForCodeInterpreter` and `UploadForFileSearch`.

```python
from dataclasses import dataclass


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class Reset:
    pass


@dataclass
class UploadForCodeInterpreter:
    file_path: str


@dataclass
class UploadForFileSearch:
    file_path: str
    vector_store_id: str
```

The `TextMessage` message type is used to communicate with the agent. It has a
`content` field that contains the message content, and a `source` field
for the sender. The `Reset` message type is a control message that resets
the memory of the agent. It has no fields. This is useful when we need to
start a new conversation with the agent.

The `UploadForCodeInterpreter` message type is used to upload data files
for the code interpreter and `UploadForFileSearch` message type is used to upload
documents for file search. Both message types have a `file_path` field that contains
the local path to the file to be uploaded.

#### Defining the Agent

Next, we define the agent class.
The agent class constructor has the following arguments: `description`,
`client`, `assistant_id`, `thread_id`, and `assistant_event_handler_factory`.
The `client` argument is the OpenAI async client object, and the
`assistant_event_handler_factory` is for creating an assistant event handler
to handle OpenAI Assistant events.
This can be used to create streaming output from the assistant.

The agent class has the following message handlers:
- `handle_message`: Handles the `TextMessage` message type, and sends back the
  response from the assistant.
- `handle_reset`: Handles the `Reset` message type, and resets the memory
    of the assistant agent.
- `handle_upload_for_code_interpreter`: Handles the `UploadForCodeInterpreter`
  message type, and uploads the file to the code interpreter.
- `handle_upload_for_file_search`: Handles the `UploadForFileSearch`
    message type, and uploads the document to the file search.


The memory of the assistant is stored inside a thread, which is kept in the
server side. The thread is referenced by the `thread_id` argument.

```python
import asyncio
import os
from typing import Any, Callable, List

import aiofiles
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from openai import AsyncAssistantEventHandler, AsyncClient
from openai.types.beta.thread import ToolResources, ToolResourcesFileSearch


class OpenAIAssistantAgent(RoutedAgent):
    """An agent implementation that uses the OpenAI Assistant API to generate
    responses.

    Args:
        description (str): The description of the agent.
        client (openai.AsyncClient): The client to use for the OpenAI API.
        assistant_id (str): The assistant ID to use for the OpenAI API.
        thread_id (str): The thread ID to use for the OpenAI API.
        assistant_event_handler_factory (Callable[[], AsyncAssistantEventHandler], optional):
            A factory function to create an async assistant event handler. Defaults to None.
            If provided, the agent will use the streaming mode with the event handler.
            If not provided, the agent will use the blocking mode to generate responses.
    """

    def __init__(
        self,
        description: str,
        client: AsyncClient,
        assistant_id: str,
        thread_id: str,
        assistant_event_handler_factory: Callable[[], AsyncAssistantEventHandler],
    ) -> None:
        super().__init__(description)
        self._client = client
        self._assistant_id = assistant_id
        self._thread_id = thread_id
        self._assistant_event_handler_factory = assistant_event_handler_factory

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> TextMessage:
        """Handle a message. This method adds the message to the thread and publishes a response."""
        # Save the message to the thread.
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.create(
                    thread_id=self._thread_id,
                    content=message.content,
                    role="user",
                    metadata={"sender": message.source},
                )
            )
        )
        # Generate a response.
        async with self._client.beta.threads.runs.stream(
            thread_id=self._thread_id,
            assistant_id=self._assistant_id,
            event_handler=self._assistant_event_handler_factory(),
        ) as stream:
            await ctx.cancellation_token.link_future(asyncio.ensure_future(stream.until_done()))

        # Get the last message.
        messages = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id, order="desc", limit=1))
        )
        last_message_content = messages.data[0].content

        # Get the text content from the last message.
        text_content = [content for content in last_message_content if content.type == "text"]
        if not text_content:
            raise ValueError(f"Expected text content in the last message: {last_message_content}")

        return TextMessage(content=text_content[0].text.value, source=self.metadata["type"])

    @message_handler()
    async def on_reset(self, message: Reset, ctx: MessageContext) -> None:
        """Handle a reset message. This method deletes all messages in the thread."""
        # Get all messages in this thread.
        all_msgs: List[str] = []
        while True:
            if not all_msgs:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id))
                )
            else:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id, after=all_msgs[-1]))
                )
            for msg in msgs.data:
                all_msgs.append(msg.id)
            if not msgs.has_next_page():
                break
        # Delete all the messages.
        for msg_id in all_msgs:
            status = await ctx.cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.messages.delete(message_id=msg_id, thread_id=self._thread_id)
                )
            )
            assert status.deleted is True

    @message_handler()
    async def on_upload_for_code_interpreter(self, message: UploadForCodeInterpreter, ctx: MessageContext) -> None:
        """Handle an upload for code interpreter. This method uploads a file and updates the thread with the file."""
        # Get the file content.
        async with aiofiles.open(message.file_path, mode="rb") as f:
            file_content = await ctx.cancellation_token.link_future(asyncio.ensure_future(f.read()))
        file_name = os.path.basename(message.file_path)
        # Upload the file.
        file = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(self._client.files.create(file=(file_name, file_content), purpose="assistants"))
        )
        # Get existing file ids from tool resources.
        thread = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(self._client.beta.threads.retrieve(thread_id=self._thread_id))
        )
        tool_resources: ToolResources = thread.tool_resources if thread.tool_resources else ToolResources()
        assert tool_resources.code_interpreter is not None
        if tool_resources.code_interpreter.file_ids:
            file_ids = tool_resources.code_interpreter.file_ids
        else:
            file_ids = [file.id]
        # Update thread with new file.
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.update(
                    thread_id=self._thread_id,
                    tool_resources={
                        "code_interpreter": {"file_ids": file_ids},
                    },
                )
            )
        )

    @message_handler()
    async def on_upload_for_file_search(self, message: UploadForFileSearch, ctx: MessageContext) -> None:
        """Handle an upload for file search. This method uploads a file and updates the vector store."""
        # Get the file content.
        async with aiofiles.open(message.file_path, mode="rb") as file:
            file_content = await ctx.cancellation_token.link_future(asyncio.ensure_future(file.read()))
        file_name = os.path.basename(message.file_path)
        # Upload the file.
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=message.vector_store_id,
                    files=[(file_name, file_content)],
                )
            )
        )
```

The agent class is a thin wrapper around the OpenAI Assistant API to implement
the message protocol. More features, such as multi-modal message handling,
can be added by extending the message protocol.

#### Assistant Event Handler

The assistant event handler provides call-backs for handling Assistant API
specific events. This is useful for handling streaming output from the assistant
and further user interface integration.

````python
from openai import AsyncAssistantEventHandler, AsyncClient
from openai.types.beta.threads import Message, Text, TextDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta
from typing_extensions import override


class EventHandler(AsyncAssistantEventHandler):
    @override
    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        print(delta.value, end="", flush=True)

    @override
    async def on_run_step_created(self, run_step: RunStep) -> None:
        details = run_step.step_details
        if details.type == "tool_calls":
            for tool in details.tool_calls:
                if tool.type == "code_interpreter":
                    print("\nGenerating code to interpret:\n\n```python")

    @override
    async def on_run_step_done(self, run_step: RunStep) -> None:
        details = run_step.step_details
        if details.type == "tool_calls":
            for tool in details.tool_calls:
                if tool.type == "code_interpreter":
                    print("\n```\nExecuting code...")

    @override
    async def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep) -> None:
        details = delta.step_details
        if details is not None and details.type == "tool_calls":
            for tool in details.tool_calls or []:
                if tool.type == "code_interpreter" and tool.code_interpreter and tool.code_interpreter.input:
                    print(tool.code_interpreter.input, end="", flush=True)

    @override
    async def on_message_created(self, message: Message) -> None:
        print(f"{'-'*80}\nAssistant:\n")

    @override
    async def on_message_done(self, message: Message) -> None:
        # print a citation to the file searched
        if not message.content:
            return
        content = message.content[0]
        if not content.type == "text":
            return
        text_content = content.text
        annotations = text_content.annotations
        citations: List[str] = []
        for index, annotation in enumerate(annotations):
            text_content.value = text_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                client = AsyncClient()
                cited_file = await client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")
        if citations:
            print("\n".join(citations))
````

#### Using the Agent

First we need to use the `openai` client to create the actual assistant,
thread, and vector store. Our AutoGen agent will be using these.

```python
import openai

# Create an assistant with code interpreter and file search tools.
oai_assistant = openai.beta.assistants.create(
    model="gpt-4o-mini",
    description="An AI assistant that helps with everyday tasks.",
    instructions="Help the user with their task.",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
)

# Create a vector store to be used for file search.
vector_store = openai.vector_stores.create()

# Create a thread which is used as the memory for the assistant.
thread = openai.beta.threads.create(
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)
```

Then, we create a runtime, and register an agent factory function for this 
agent with the runtime.

```python
from autogen_core import SingleThreadedAgentRuntime

runtime = SingleThreadedAgentRuntime()
await OpenAIAssistantAgent.register(
    runtime,
    "assistant",
    lambda: OpenAIAssistantAgent(
        description="OpenAI Assistant Agent",
        client=openai.AsyncClient(),
        assistant_id=oai_assistant.id,
        thread_id=thread.id,
        assistant_event_handler_factory=lambda: EventHandler(),
    ),
)
agent = AgentId("assistant", "default")
```

Let's turn on logging to see what's happening under the hood.

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_core").setLevel(logging.DEBUG)
```

Let's send a greeting message to the agent, and see the response streamed back.

```python
runtime.start()
await runtime.send_message(TextMessage(content="Hello, how are you today!", source="user"), agent)
await runtime.stop_when_idle()
```

```text
INFO:autogen_core:Sending message of type TextMessage to assistant: {'content': 'Hello, how are you today!', 'source': 'user'}
INFO:autogen_core:Calling message handler for assistant:default with message type TextMessage sent by Unknown
--------------------------------------------------------------------------------
Assistant:

Hello! I'm here and ready to assist you. How can I help you today?INFO:autogen_core:Resolving response with message type TextMessage for recipient None from assistant: {'content': "Hello! I'm here and ready to assist you. How can I help you today?", 'source': 'assistant'}
```

#### Assistant with Code Interpreter

Let's ask some math question to the agent, and see it uses the code interpreter
to answer the question.

```python
runtime.start()
await runtime.send_message(TextMessage(content="What is 1332322 x 123212?", source="user"), agent)
await runtime.stop_when_idle()
```

````text
INFO:autogen_core:Sending message of type TextMessage to assistant: {'content': 'What is 1332322 x 123212?', 'source': 'user'}
INFO:autogen_core:Calling message handler for assistant:default with message type TextMessage sent by Unknown
# Calculating the product of 1332322 and 123212
result = 1332322 * 123212
result
```
Executing code...
--------------------------------------------------------------------------------
Assistant:

The product of 1,332,322 and 123,212 is 164,158,058,264.INFO:autogen_core:Resolving response with message type TextMessage for recipient None from assistant: {'content': 'The product of 1,332,322 and 123,212 is 164,158,058,264.', 'source': 'assistant'}
````

Let's get some data from Seattle Open Data portal. We will be using the
[City of Seattle Wage Data](https://data.seattle.gov/City-Business/City-of-Seattle-Wage-Data/2khk-5ukd/).
Let's download it first.

```python
import requests

response = requests.get("https://data.seattle.gov/resource/2khk-5ukd.csv")
with open("seattle_city_wages.csv", "wb") as file:
    file.write(response.content)
```

Let's send the file to the agent using an `UploadForCodeInterpreter` message.

```python
runtime.start()
await runtime.send_message(UploadForCodeInterpreter(file_path="seattle_city_wages.csv"), agent)
await runtime.stop_when_idle()
```

```text
INFO:autogen_core:Sending message of type UploadForCodeInterpreter to assistant: {'file_path': 'seattle_city_wages.csv'}
INFO:autogen_core:Calling message handler for assistant:default with message type UploadForCodeInterpreter sent by Unknown
INFO:autogen_core:Resolving response with message type NoneType for recipient None from assistant: None
```

We can now ask some questions about the data to the agent.

```python
runtime.start()
await runtime.send_message(TextMessage(content="Take a look at the uploaded CSV file.", source="user"), agent)
await runtime.stop_when_idle()
```

````text
INFO:autogen_core:Sending message of type TextMessage to assistant: {'content': 'Take a look at the uploaded CSV file.', 'source': 'user'}
INFO:autogen_core:Calling message handler for assistant:default with message type TextMessage sent by Unknown
import pandas as pd

# Load the uploaded CSV file to examine its contents
file_path = '/mnt/data/file-oEvRiyGyHc2jZViKyDqL8aoh'
csv_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
csv_data.head()
```
Executing code...
--------------------------------------------------------------------------------
Assistant:

The uploaded CSV file contains the following columns:

1. **department**: The department in which the individual works.
2. **last_name**: The last name of the employee.
3. **first_name**: The first name of the employee.
4. **job_title**: The job title of the employee.
5. **hourly_rate**: The hourly rate for the employee's position.

Here are the first few entries from the file:

| department                     | last_name | first_name | job_title                          | hourly_rate |
|--------------------------------|-----------|------------|------------------------------------|-------------|
| Police Department              | Aagard    | Lori       | Pol Capt-Precinct                 | 112.70      |
| Police Department              | Aakervik  | Dag        | Pol Ofcr-Detective                | 75.61       |
| Seattle City Light             | Aaltonen  | Evan       | Pwrline Clear Tree Trimmer        | 53.06       |
| Seattle Public Utilities       | Aar       | Abdimallik | Civil Engrng Spec,Sr               | 64.43       |
| Seattle Dept of Transportation | Abad      | Abigail    | Admin Spec II-BU                  | 37.40       |

If you need any specific analysis or information from this data, please let me know!INFO:autogen_core:Resolving response with message type TextMessage for recipient None from assistant: {'content': "The uploaded CSV file contains the following columns:\n\n1. **department**: The department in which the individual works.\n2. **last_name**: The last name of the employee.\n3. **first_name**: The first name of the employee.\n4. **job_title**: The job title of the employee.\n5. **hourly_rate**: The hourly rate for the employee's position.\n\nHere are the first few entries from the file:\n\n| department                     | last_name | first_name | job_title                          | hourly_rate |\n|--------------------------------|-----------|------------|------------------------------------|-------------|\n| Police Department              | Aagard    | Lori       | Pol Capt-Precinct                 | 112.70      |\n| Police Department              | Aakervik  | Dag        | Pol Ofcr-Detective                | 75.61       |\n| Seattle City Light             | Aaltonen  | Evan       | Pwrline Clear Tree Trimmer        | 53.06       |\n| Seattle Public Utilities       | Aar       | Abdimallik | Civil Engrng Spec,Sr               | 64.43       |\n| Seattle Dept of Transportation | Abad      | Abigail    | Admin Spec II-BU                  | 37.40       |\n\nIf you need any specific analysis or information from this data, please let me know!", 'source': 'assistant'}
````

```python
runtime.start()
await runtime.send_message(TextMessage(content="What are the top-10 salaries?", source="user"), agent)
await runtime.stop_when_idle()
```

````text
INFO:autogen_core:Sending message of type TextMessage to assistant: {'content': 'What are the top-10 salaries?', 'source': 'user'}
INFO:autogen_core:Calling message handler for assistant:default with message type TextMessage sent by Unknown
# Sorting the data by hourly_rate in descending order and selecting the top 10 salaries
top_10_salaries = csv_data[['first_name', 'last_name', 'job_title', 'hourly_rate']].sort_values(by='hourly_rate', ascending=False).head(10)
top_10_salaries.reset_index(drop=True, inplace=True)
top_10_salaries
```
Executing code...
--------------------------------------------------------------------------------
Assistant:

Here are the top 10 salaries based on the hourly rates from the CSV file:

| First Name | Last Name | Job Title                          | Hourly Rate |
|------------|-----------|------------------------------------|-------------|
| Eric       | Barden    | Executive4                        | 139.61      |
| Idris      | Beauregard| Executive3                        | 115.90      |
| Lori       | Aagard    | Pol Capt-Precinct                 | 112.70      |
| Krista     | Bair      | Pol Capt-Precinct                 | 108.74      |
| Amy        | Bannister | Fire Chief, Dep Adm-80 Hrs        | 104.07      |
| Ginger     | Armbruster| Executive2                        | 102.42      |
| William    | Andersen  | Executive2                        | 102.42      |
| Valarie    | Anderson  | Executive2                        | 102.42      |
| Paige      | Alderete  | Executive2                        | 102.42      |
| Kathryn    | Aisenberg | Executive2                        | 100.65      |

If you need any further details or analysis, let me know!INFO:autogen_core:Resolving response with message type TextMessage for recipient None from assistant: {'content': 'Here are the top 10 salaries based on the hourly rates from the CSV file:\n\n| First Name | Last Name | Job Title                          | Hourly Rate |\n|------------|-----------|------------------------------------|-------------|\n| Eric       | Barden    | Executive4                        | 139.61      |\n| Idris      | Beauregard| Executive3                        | 115.90      |\n| Lori       | Aagard    | Pol Capt-Precinct                 | 112.70      |\n| Krista     | Bair      | Pol Capt-Precinct                 | 108.74      |\n| Amy        | Bannister | Fire Chief, Dep Adm-80 Hrs        | 104.07      |\n| Ginger     | Armbruster| Executive2                        | 102.42      |\n| William    | Andersen  | Executive2                        | 102.42      |\n| Valarie    | Anderson  | Executive2                        | 102.42      |\n| Paige      | Alderete  | Executive2                        | 102.42      |\n| Kathryn    | Aisenberg | Executive2                        | 100.65      |\n\nIf you need any further details or analysis, let me know!', 'source': 'assistant'}
````

#### Assistant with File Search

Let's try the Q&A over document feature. We first download Wikipedia page
on the Third Anglo-Afghan War.

```python
response = requests.get("https://en.wikipedia.org/wiki/Third_Anglo-Afghan_War")
with open("third_anglo_afghan_war.html", "wb") as file:
    file.write(response.content)
```

Send the file to the agent using an `UploadForFileSearch` message.

```python
runtime.start()
await runtime.send_message(
    UploadForFileSearch(file_path="third_anglo_afghan_war.html", vector_store_id=vector_store.id), agent
)
await runtime.stop_when_idle()
```

```text
INFO:autogen_core:Sending message of type UploadForFileSearch to assistant: {'file_path': 'third_anglo_afghan_war.html', 'vector_store_id': 'vs_h3xxPbJFnd1iZ9WdjsQwNdrp'}
INFO:autogen_core:Calling message handler for assistant:default with message type UploadForFileSearch sent by Unknown
INFO:autogen_core:Resolving response with message type NoneType for recipient None from assistant: None
```

Let's ask some questions about the document to the agent. Before asking,
we reset the agent memory to start a new conversation.

```python
runtime.start()
await runtime.send_message(Reset(), agent)
await runtime.send_message(
    TextMessage(
        content="When and where was the treaty of Rawalpindi signed? Answer using the document provided.", source="user"
    ),
    agent,
)
await runtime.stop_when_idle()
```

```text
INFO:autogen_core:Sending message of type Reset to assistant: {}
INFO:autogen_core:Calling message handler for assistant:default with message type Reset sent by Unknown
INFO:autogen_core:Resolving response with message type NoneType for recipient None from assistant: None
INFO:autogen_core:Sending message of type TextMessage to assistant: {'content': 'When and where was the treaty of Rawalpindi signed? Answer using the document provided.', 'source': 'user'}
INFO:autogen_core:Calling message handler for assistant:default with message type TextMessage sent by Unknown
--------------------------------------------------------------------------------
Assistant:

The Treaty of Rawalpindi was signed on **8 August 1919**. The location of the signing was in **Rawalpindi**, which is in present-day Pakistan【6:0†source】.INFO:autogen_core:Resolving response with message type TextMessage for recipient None from assistant: {'content': 'The Treaty of Rawalpindi was signed on **8 August 1919**. The location of the signing was in **Rawalpindi**, which is in present-day Pakistan【6:0†source】.', 'source': 'assistant'}
[0] third_anglo_afghan_war.html
```

That's it! We have successfully built an agent backed by OpenAI Assistant.


## Using LangGraph-Backed Agent

*[cookbook/langgraph-agent.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/langgraph-agent.html)*

### Using LangGraph-Backed Agent

This example demonstrates how to create an AI agent using LangGraph.
Based on the example in the LangGraph documentation:
https://langchain-ai.github.io/langgraph/.

First install the dependencies:

```python
# pip install langgraph langchain-openai azure-identity
```

Let's import the modules.

```python
from dataclasses import dataclass
from typing import Any, Callable, List, Literal

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool  # pyright: ignore
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
```

Define our message type that will be used to communicate with the agent.

```python
@dataclass
class Message:
    content: str
```

Define the tools the agent will use.

```python
@tool  # pyright: ignore
def get_weather(location: str) -> str:
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."
```

Define the agent using LangGraph's API.

```python
class LangGraphToolUseAgent(RoutedAgent):
    def __init__(self, description: str, model: ChatOpenAI, tools: List[Callable[..., Any]]) -> None:  # pyright: ignore
        super().__init__(description)
        self._model = model.bind_tools(tools)  # pyright: ignore

        # Define the function that determines whether to continue or not
        def should_continue(state: MessagesState) -> Literal["tools", END]:  # type: ignore
            messages = state["messages"]
            last_message = messages[-1]
            # If the LLM makes a tool call, then we route to the "tools" node
            if last_message.tool_calls:  # type: ignore
                return "tools"
            # Otherwise, we stop (reply to the user)
            return END

        # Define the function that calls the model
        async def call_model(state: MessagesState):  # type: ignore
            messages = state["messages"]
            response = await self._model.ainvoke(messages)
            # We return a list, because this will get added to the existing list
            return {"messages": [response]}

        tool_node = ToolNode(tools)  # pyright: ignore

        # Define a new graph
        self._workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        self._workflow.add_node("agent", call_model)  # pyright: ignore
        self._workflow.add_node("tools", tool_node)  # pyright: ignore

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self._workflow.set_entry_point("agent")

        # We now add a conditional edge
        self._workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,  # type: ignore
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        self._workflow.add_edge("tools", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable.
        # Note that we're (optionally) passing the memory when compiling the graph
        self._app = self._workflow.compile()

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Use the Runnable
        final_state = await self._app.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content="You are a helpful AI assistant. You can use tools to help answer questions."
                    ),
                    HumanMessage(content=message.content),
                ]
            },
            config={"configurable": {"thread_id": 42}},
        )
        response = Message(content=final_state["messages"][-1].content)
        return response
```

Now let's test the agent. First we need to create an agent runtime and
register the agent, by providing the agent's name and a factory function
that will create the agent.

```python
runtime = SingleThreadedAgentRuntime()
await LangGraphToolUseAgent.register(
    runtime,
    "langgraph_tool_use_agent",
    lambda: LangGraphToolUseAgent(
        "Tool use agent",
        ChatOpenAI(
            model="gpt-4o",
            # api_key=os.getenv("OPENAI_API_KEY"),
        ),
        # AzureChatOpenAI(
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        #     # Using Azure Active Directory authentication.
        #     azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential()),
        #     # Using API key.
        #     # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        # ),
        [get_weather],
    ),
)
agent = AgentId("langgraph_tool_use_agent", key="default")
```

Start the agent runtime.

```python
runtime.start()
```

Send a direct message to the agent, and print the response.

```python
response = await runtime.send_message(Message("What's the weather in SF?"), agent)
print(response.content)
```

```text
The current weather in San Francisco is 60 degrees and foggy.
```

Stop the agent runtime.

```python
await runtime.stop()
```


## Using LlamaIndex-Backed Agent

*[cookbook/llamaindex-agent.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/llamaindex-agent.html)*

### Using LlamaIndex-Backed Agent

This example demonstrates how to create an AI agent using LlamaIndex.

First install the dependencies:

```python
# pip install "llama-index-readers-web" "llama-index-readers-wikipedia" "llama-index-tools-wikipedia" "llama-index-embeddings-azure-openai" "llama-index-llms-azure-openai" "llama-index" "azure-identity"
```

Let's import the modules.

```python
import os
from typing import List, Optional

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.tools.wikipedia import WikipediaToolSpec
from pydantic import BaseModel
```

Define our message type that will be used to communicate with the agent.

```python
class Resource(BaseModel):
    content: str
    node_id: str
    score: Optional[float] = None


class Message(BaseModel):
    content: str
    sources: Optional[List[Resource]] = None
```

Define the agent using LLamaIndex's API.

```python
class LlamaIndexAgent(RoutedAgent):
    def __init__(self, description: str, llama_index_agent: AgentRunner, memory: BaseMemory | None = None) -> None:
        super().__init__(description)

        self._llama_index_agent = llama_index_agent
        self._memory = memory

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # retriever history messages from memory!
        history_messages: List[ChatMessage] = []

        response: AgentChatResponse  # pyright: ignore
        if self._memory is not None:
            history_messages = self._memory.get(input=message.content)

            response = await self._llama_index_agent.achat(message=message.content, history_messages=history_messages)  # pyright: ignore
        else:
            response = await self._llama_index_agent.achat(message=message.content)  # pyright: ignore

        if isinstance(response, AgentChatResponse):
            if self._memory is not None:
                self._memory.put(ChatMessage(role=MessageRole.USER, content=message.content))
                self._memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response.response))

            assert isinstance(response.response, str)

            resources: List[Resource] = [
                Resource(content=source_node.get_text(), score=source_node.score, node_id=source_node.id_)
                for source_node in response.source_nodes
            ]

            tools: List[Resource] = [
                Resource(content=source.content, node_id=source.tool_name) for source in response.sources
            ]

            resources.extend(tools)
            return Message(content=response.response, sources=resources)
        else:
            return Message(content="I'm sorry, I don't have an answer for you.")
```

Setting up LlamaIndex.

```python
# llm = AzureOpenAI(
#     deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     temperature=0.0,
#     azure_ad_token_provider = get_bearer_token_provider(DefaultAzureCredential()),
#     # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )
llm = OpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# embed_model = AzureOpenAIEmbedding(
#     deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
#     temperature=0.0,
#     azure_ad_token_provider = get_bearer_token_provider(DefaultAzureCredential()),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
# )
embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY"),
)

Settings.llm = llm
Settings.embed_model = embed_model
```

Create the tools.

```python
wiki_spec = WikipediaToolSpec()
wikipedia_tool = wiki_spec.to_tool_list()[1]
```

Now let's test the agent. First we need to create an agent runtime and
register the agent, by providing the agent's name and a factory function
that will create the agent.

```python
runtime = SingleThreadedAgentRuntime()
await LlamaIndexAgent.register(
    runtime,
    "chat_agent",
    lambda: LlamaIndexAgent(
        description="Llama Index Agent",
        llama_index_agent=ReActAgent.from_tools(
            tools=[wikipedia_tool],
            llm=llm,
            max_iterations=8,
            memory=ChatSummaryMemoryBuffer(llm=llm, token_limit=16000),
            verbose=True,
        ),
    ),
)
agent = AgentId("chat_agent", "default")
```

Start the agent runtime.

```python
runtime.start()
```

Send a direct message to the agent, and print the response.

```python
message = Message(content="What are the best movies from studio Ghibli?")
response = await runtime.send_message(message, agent)
assert isinstance(response, Message)
print(response.content)
```

```text
> Running step 3cbf60cd-9827-4dfe-a3a9-eaff2bed9b75. Step input: What are the best movies from studio Ghibli?
[1;3;38;5;200mThought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: search_data
Action Input: {'query': 'best movies from Studio Ghibli'}
[0m[1;3;34mObservation: This is a list of works (films, television, shorts etc.) by the Japanese animation studio Studio Ghibli.


== Works ==


=== Feature films ===


=== Television ===


=== Short films ===

These are short films, including those created for television, theatrical release, and the Ghibli Museum. Original video animation releases and music videos (theatrical and television) are also listed in this section.


=== Commercials ===


=== Video games ===


=== Stage productions ===
Princess Mononoke (2013)
Nausicaä of the Valley of the Wind (2019)
Spirited Away (2022)
My Neighbour Totoro (2022)


=== Other works ===
The works listed here consist of works that do not fall into the above categories. All of these films have been released on DVD or Blu-ray in Japan as part of the Ghibli Gakujutsu Library.


=== Exhibitions ===
A selection of layout designs for animated productions was exhibited in the Studio Ghibli Layout Designs: Understanding the Secrets of Takahata and Miyazaki Animation exhibition tour, which started in the Museum of Contemporary Art Tokyo (July 28, 2008 to September 28, 2008) and subsequently travelled to different museums throughout Japan and Asia, concluding its tour of Japan in the Fukuoka Asian Art Museum (October 12, 2013 to January 26, 2014) and its tour of Asia in the Hong Kong Heritage Museum (May 14, 2014 to August 31, 2014). Between October 4, 2014 and March 1, 2015 the layout designs were exhibited at Art Ludique in Paris. The exhibition catalogues contain annotated reproductions of the displayed artwork.


== Related works ==
These works were not created by Studio Ghibli, but were produced by a variety of studios and people who went on to form or join Studio Ghibli. This includes members of Topcraft that went on to create Studio Ghibli in 1985; works produced by Toei Animation, TMS Entertainment, Nippon Animation or other studios and featuring involvement by Hayao Miyazaki, Isao Takahata or other Ghibli staffers. The list also includes works created in cooperation with Studio Ghibli.


=== Pre-Ghibli ===


=== Cooperative works ===


=== Distributive works ===
These Western animated films (plus one Japanese film) have been distributed by Studio Ghibli, and now through their label, Ghibli Museum Library.


=== Contributive works ===
Studio Ghibli has made contributions to the following anime series and movies:


== Significant achievements ==
The highest-grossing film of 1989 in Japan: Kiki's Delivery Service
The highest-grossing film of 1991 in Japan: Only Yesterday
The highest-grossing film of 1992 in Japan: Porco Rosso
The highest-grossing film of 1994 in Japan: Pom Poko
The highest-grossing film of 1995 in Japan; the first Japanese film in Dolby Digital: Whisper of the Heart
The highest-grossing film of 2002 in Japan: Spirited Away
The highest-grossing film of 2008 in Japan: Ponyo
The highest-grossing Japanese film of 2010 in Japan: The Secret World of Arrietty
The highest-grossing film of 2013 in Japan: The Wind Rises
The first Studio Ghibli film to use computer graphics: Pom Poko
The first Miyazaki feature to use computer graphics, and the first Studio Ghibli film to use digital coloring; the first animated feature in Japan's history to gross more than 10 billion yen at the box office and the first animated film ever to win a National Academy Award for Best Picture of the Year: Princess Mononoke
The first Studio Ghibli film to be shot using a 100% digital process: My Neighbors the Yamadas
The first Miyazaki feature to be shot using a 100% digital process; the first film to gross $200 million worldwide before opening in North America; the film to finally overtake Titanic at the Japanese box office, becoming the top-grossing film in the history of Japanese cinema: Spirited Away
The first anime and traditionally animated winner of the Academy Award for Best Animated Feature: Spirited Away at the 75th Academy Awards. They would later win this award for a second time with The Boy and the Heron at the 96th Academy Awards, marking the second time a traditionally animated film won the award.


== Notes ==


== References ==
[0m> Running step 561e3dd3-d98b-4d37-b612-c99387182ee0. Step input: None
[1;3;38;5;200mThought: I can answer without using any more tools. I'll use the user's language to answer.
Answer: Studio Ghibli has produced many acclaimed films over the years. Some of the best and most popular movies from Studio Ghibli include:

1. **Spirited Away (2001)** - Directed by Hayao Miyazaki, this film won the Academy Award for Best Animated Feature and is one of the highest-grossing films in Japanese history.
2. **My Neighbor Totoro (1988)** - Another classic by Hayao Miyazaki, this film is beloved for its heartwarming story and iconic characters.
3. **Princess Mononoke (1997)** - This epic fantasy film, also directed by Miyazaki, is known for its complex themes and stunning animation.
4. **Howl's Moving Castle (2004)** - Based on the novel by Diana Wynne Jones, this film features a magical story and beautiful animation.
5. **Kiki's Delivery Service (1989)** - A charming coming-of-age story about a young witch starting her own delivery service.
6. **Grave of the Fireflies (1988)** - Directed by Isao Takahata, this poignant film is a heartbreaking tale of two siblings struggling to survive during World War II.
7. **Ponyo (2008)** - A delightful and visually stunning film about a young fish-girl who wants to become human.
8. **The Wind Rises (2013)** - A more mature film by Miyazaki, focusing on the life of an aircraft designer during wartime Japan.
9. **The Secret World of Arrietty (2010)** - Based on Mary Norton's novel "The Borrowers
```

```python
if response.sources is not None:
    for source in response.sources:
        print(source.content)
```

```text
This is a list of works (films, television, shorts etc.) by the Japanese animation studio Studio Ghibli.


== Works ==


=== Feature films ===


=== Television ===


=== Short films ===

These are short films, including those created for television, theatrical release, and the Ghibli Museum. Original video animation releases and music videos (theatrical and television) are also listed in this section.


=== Commercials ===


=== Video games ===


=== Stage productions ===
Princess Mononoke (2013)
Nausicaä of the Valley of the Wind (2019)
Spirited Away (2022)
My Neighbour Totoro (2022)


=== Other works ===
The works listed here consist of works that do not fall into the above categories. All of these films have been released on DVD or Blu-ray in Japan as part of the Ghibli Gakujutsu Library.


=== Exhibitions ===
A selection of layout designs for animated productions was exhibited in the Studio Ghibli Layout Designs: Understanding the Secrets of Takahata and Miyazaki Animation exhibition tour, which started in the Museum of Contemporary Art Tokyo (July 28, 2008 to September 28, 2008) and subsequently travelled to different museums throughout Japan and Asia, concluding its tour of Japan in the Fukuoka Asian Art Museum (October 12, 2013 to January 26, 2014) and its tour of Asia in the Hong Kong Heritage Museum (May 14, 2014 to August 31, 2014). Between October 4, 2014 and March 1, 2015 the layout designs were exhibited at Art Ludique in Paris. The exhibition catalogues contain annotated reproductions of the displayed artwork.


== Related works ==
These works were not created by Studio Ghibli, but were produced by a variety of studios and people who went on to form or join Studio Ghibli. This includes members of Topcraft that went on to create Studio Ghibli in 1985; works produced by Toei Animation, TMS Entertainment, Nippon Animation or other studios and featuring involvement by Hayao Miyazaki, Isao Takahata or other Ghibli staffers. The list also includes works created in cooperation with Studio Ghibli.


=== Pre-Ghibli ===


=== Cooperative works ===


=== Distributive works ===
These Western animated films (plus one Japanese film) have been distributed by Studio Ghibli, and now through their label, Ghibli Museum Library.


=== Contributive works ===
Studio Ghibli has made contributions to the following anime series and movies:


== Significant achievements ==
The highest-grossing film of 1989 in Japan: Kiki's Delivery Service
The highest-grossing film of 1991 in Japan: Only Yesterday
The highest-grossing film of 1992 in Japan: Porco Rosso
The highest-grossing film of 1994 in Japan: Pom Poko
The highest-grossing film of 1995 in Japan; the first Japanese film in Dolby Digital: Whisper of the Heart
The highest-grossing film of 2002 in Japan: Spirited Away
The highest-grossing film of 2008 in Japan: Ponyo
The highest-grossing Japanese film of 2010 in Japan: The Secret World of Arrietty
The highest-grossing film of 2013 in Japan: The Wind Rises
The first Studio Ghibli film to use computer graphics: Pom Poko
The first Miyazaki feature to use computer graphics, and the first Studio Ghibli film to use digital coloring; the first animated feature in Japan's history to gross more than 10 billion yen at the box office and the first animated film ever to win a National Academy Award for Best Picture of the Year: Princess Mononoke
The first Studio Ghibli film to be shot using a 100% digital process: My Neighbors the Yamadas
The first Miyazaki feature to be shot using a 100% digital process; the first film to gross $200 million worldwide before opening in North America; the film to finally overtake Titanic at the Japanese box office, becoming the top-grossing film in the history of Japanese cinema: Spirited Away
The first anime and traditionally animated winner of the Academy Award for Best Animated Feature: Spirited Away at the 75th Academy Awards. They would later win this award for a second time with The Boy and the Heron at the 96th Academy Awards, marking the second time a traditionally animated film won the award.


== Notes ==


== References ==
```

Stop the agent runtime.

```python
await runtime.stop()
```


## Local LLMs with LiteLLM & Ollama

*[cookbook/local-llms-ollama-litellm.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/local-llms-ollama-litellm.html)*

### Local LLMs with LiteLLM & Ollama

In this notebook we'll create two agents, Joe and Cathy who like to tell jokes to each other. The agents will use locally running LLMs.

Follow the guide at https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama/ to understand how to install LiteLLM and Ollama.

We encourage going through the link, but if you're in a hurry and using Linux, run these:  
  
```
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:1b

pip install 'litellm[proxy]'
litellm --model ollama/llama3.2:1b
```  

This will run the proxy server and it will be available at 'http://0.0.0.0:4000/'.

To get started, let's import some classes.

```python
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

Set up out local LLM model client.

```python
def get_model_client() -> OpenAIChatCompletionClient:  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model="llama3.2:1b",
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://0.0.0.0:4000",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )
```

Define a simple message class

```python
@dataclass
class Message:
    content: str
```

Now, the Agent.

We define the role of the Agent using the `SystemMessage` and set up a condition for termination.

```python
@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self.name = name
        self.count = 0
        self._system_messages = [
            SystemMessage(
                content=f"Your name is {name} and you are a part of a duo of comedians."
                "You laugh when you find the joke funny, else reply 'I need to go now'.",
            )
        ]
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        self.count += 1
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._system_messages + await self._model_context.get_messages())

        print(f"\n{self.name}: {message.content}")

        if "I need to go".lower() in message.content.lower() or self.count > 2:
            return

        await self._model_context.add_message(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
        await self.publish_message(Message(content=result.content), DefaultTopicId())  # type: ignore
```

Set up the agents.

```python
runtime = SingleThreadedAgentRuntime()

model_client = get_model_client()

cathy = await Assistant.register(
    runtime,
    "cathy",
    lambda: Assistant(name="Cathy", model_client=model_client),
)

joe = await Assistant.register(
    runtime,
    "joe",
    lambda: Assistant(name="Joe", model_client=model_client),
)
```

Let's run everything!

```python
runtime.start()
await runtime.send_message(
    Message("Joe, tell me a joke."),
    recipient=AgentId(joe, "default"),
    sender=AgentId(cathy, "default"),
)
await runtime.stop_when_idle()

# Close the connections to the model clients.
await model_client.close()
```

```text
/tmp/ipykernel_1417357/2124203426.py:22: UserWarning: Resolved model mismatch: gpt-4o-2024-05-13 != ollama/llama3.1:8b. Model mapping may be incorrect.
  result = await self._model_client.create(self._system_messages + await self._model_context.get_messages())

Joe: Joe, tell me a joke.

Cathy: Here's one:

Why couldn't the bicycle stand up by itself?

(waiting for your reaction...)

Joe: *laughs* It's because it was two-tired! Ahahaha! That's a good one! I love it!

Cathy: *roars with laughter* HAHAHAHA! Oh man, that's a classic! I'm glad you liked it! The setup is perfect and the punchline is just... *chuckles* Two-tired! I mean, come on! That's genius! We should definitely add that one to our act!

Joe: I need to go now.
```


## Instrumentating your code locally

*[cookbook/instrumenting.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/instrumenting.html)*

### Instrumentating your code locally

AutoGen supports instrumenting your code using [OpenTelemetry](https://opentelemetry.io). This allows you to collect traces and logs from your code and send them to a backend of your choice.

While debugging, you can use a local backend such as [Aspire](https://aspiredashboard.com/) or [Jaeger](https://www.jaegertracing.io/). In this guide we will use Aspire as an example.

#### Setting up Aspire

Follow the instructions [here](https://learn.microsoft.com/en-us/dotnet/aspire/fundamentals/dashboard/overview?tabs=bash#standalone-mode) to set up Aspire in standalone mode. This will require Docker to be installed on your machine.

#### Instrumenting your code

Once you have a dashboard set up, now it's a matter of sending traces and logs to it. You can follow the steps in the [Telemetry Guide](../framework/telemetry.md) to set up the opentelemetry sdk and exporter.

After instrumenting your code with the Aspire Dashboard running, you should see traces and logs appear in the dashboard as your code runs.

#### Observing LLM calls using Open AI

If you are using the Open AI package, you can observe the LLM calls by setting up the opentelemetry for that library. We use [opentelemetry-instrumentation-openai](https://pypi.org/project/opentelemetry-instrumentation-openai/) in this example.

Install the package:
```bash
pip install opentelemetry-instrumentation-openai
```

Enable the instrumentation:
```python
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument()
```

Now running your code will send traces including the LLM calls to your telemetry backend (Aspire in our case).

![Open AI Telemetry logs](../../../images/open-ai-telemetry-example.png)


## Topic Subscription Scenarios

*[cookbook/topic-subscription-scenarios.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/topic-subscription-scenarios.html)*

#### Topic and Subscription Example Scenarios

##### Introduction

In this cookbook, we explore how broadcasting works for agent communication in AutoGen using four different broadcasting scenarios. These scenarios illustrate various ways to handle and distribute messages among agents. We'll use a consistent example of a tax management company processing client requests to demonstrate each scenario.

##### Scenario Overview

Imagine a tax management company that offers various services to clients, such as tax planning, dispute resolution, compliance, and preparation. The company employs a team of tax specialists, each with expertise in one of these areas, and a tax system manager who oversees the operations.

Clients submit requests that need to be processed by the appropriate specialists. The communication between the clients, the tax system manager, and the tax specialists is handled through broadcasting in this system.

We'll explore how different broadcasting scenarios affect the way messages are distributed among agents and how they can be used to tailor the communication flow to specific needs.

---

##### Broadcasting Scenarios Overview

We will cover the following broadcasting scenarios:

1. **Single-Tenant, Single Scope of Publishing**
2. **Multi-Tenant, Single Scope of Publishing**
3. **Single-Tenant, Multiple Scopes of Publishing**
4. **Multi-Tenant, Multiple Scopes of Publishing**


Each scenario represents a different approach to message distribution and agent interaction within the system. By understanding these scenarios, you can design agent communication strategies that best fit your application's requirements.

```python
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core._default_subscription import DefaultSubscription
from autogen_core._default_topic import DefaultTopicId
from autogen_core.models import (
    SystemMessage,
)
```

```python
class TaxSpecialty(str, Enum):
    PLANNING = "planning"
    DISPUTE_RESOLUTION = "dispute_resolution"
    COMPLIANCE = "compliance"
    PREPARATION = "preparation"


@dataclass
class ClientRequest:
    content: str


@dataclass
class RequestAssessment:
    content: str


class TaxSpecialist(RoutedAgent):
    def __init__(
        self,
        description: str,
        specialty: TaxSpecialty,
        system_messages: List[SystemMessage],
    ) -> None:
        super().__init__(description)
        self.specialty = specialty
        self._system_messages = system_messages
        self._memory: List[ClientRequest] = []

    @message_handler
    async def handle_message(self, message: ClientRequest, ctx: MessageContext) -> None:
        # Process the client request.
        print(f"\n{'='*50}\nTax specialist {self.id} with specialty {self.specialty}:\n{message.content}")
        # Send a response back to the manager
        if ctx.topic_id is None:
            raise ValueError("Topic ID is required for broadcasting")
        await self.publish_message(
            message=RequestAssessment(content=f"I can handle this request in {self.specialty}."),
            topic_id=ctx.topic_id,
        )
```

##### 1. Single-Tenant, Single Scope of Publishing

###### Scenarios Explanation
In the single-tenant, single scope of publishing scenario:

- All agents operate within a single tenant (e.g., one client or user session).
- Messages are published to a single topic, and all agents subscribe to this topic.
- Every agent receives every message that gets published to the topic.

This scenario is suitable for situations where all agents need to be aware of all messages, and there's no need to isolate communication between different groups of agents or sessions.

###### Application in the Tax Specialist Company

In our tax specialist company, this scenario implies:

- All tax specialists receive every client request and internal message.
- All agents collaborate closely, with full visibility of all communications.
- Useful for tasks or teams where all agents need to be aware of all messages.

###### How the Scenario Works

- Subscriptions: All agents use the default subscription(e.g., "default").
- Publishing: Messages are published to the default topic.
- Message Handling: Each agent decides whether to act on a message based on its content and available handlers.

###### Benefits
- Simplicity: Easy to set up and understand.
- Collaboration: Promotes transparency and collaboration among agents.
- Flexibility: Agents can dynamically decide which messages to process.

###### Considerations
- Scalability: May not scale well with a large number of agents or messages.
- Efficiency: Agents may receive many irrelevant messages, leading to unnecessary processing.

```python
async def run_single_tenant_single_scope() -> None:
    # Create the runtime.
    runtime = SingleThreadedAgentRuntime()

    # Register TaxSpecialist agents for each specialty
    specialist_agent_type_1 = "TaxSpecialist_1"
    specialist_agent_type_2 = "TaxSpecialist_2"
    await TaxSpecialist.register(
        runtime=runtime,
        type=specialist_agent_type_1,
        factory=lambda: TaxSpecialist(
            description="A tax specialist 1",
            specialty=TaxSpecialty.PLANNING,
            system_messages=[SystemMessage(content="You are a tax specialist.")],
        ),
    )

    await TaxSpecialist.register(
        runtime=runtime,
        type=specialist_agent_type_2,
        factory=lambda: TaxSpecialist(
            description="A tax specialist 2",
            specialty=TaxSpecialty.DISPUTE_RESOLUTION,
            system_messages=[SystemMessage(content="You are a tax specialist.")],
        ),
    )

    # Add default subscriptions for each agent type
    await runtime.add_subscription(DefaultSubscription(agent_type=specialist_agent_type_1))
    await runtime.add_subscription(DefaultSubscription(agent_type=specialist_agent_type_2))

    # Start the runtime and send a message to agents on default topic
    runtime.start()
    await runtime.publish_message(ClientRequest("I need to have my tax for 2024 prepared."), topic_id=DefaultTopicId())
    await runtime.stop_when_idle()


await run_single_tenant_single_scope()
```

```text
==================================================
Tax specialist TaxSpecialist_1:default with specialty TaxSpecialty.PLANNING:
I need to have my tax for 2024 prepared.

==================================================
Tax specialist TaxSpecialist_2:default with specialty TaxSpecialty.DISPUTE_RESOLUTION:
I need to have my tax for 2024 prepared.
```

##### 2. Multi-Tenant, Single Scope of Publishing

###### Scenario Explanation

In the multi-tenant, single scope of publishing scenario:

- There are multiple tenants (e.g., multiple clients or user sessions).
- Each tenant has its own isolated topic through the topic source.
- All agents within a tenant subscribe to the tenant's topic. If needed, new agent instances are created for each tenant.
- Messages are only visible to agents within the same tenant.

This scenario is useful when you need to isolate communication between different tenants but want all agents within a tenant to be aware of all messages.

###### Application in the Tax Specialist Company

In this scenario:

- The company serves multiple clients (tenants) simultaneously.
- For each client, a dedicated set of agent instances is created.
- Each client's communication is isolated from others.
- All agents for a client receive messages published to that client's topic.

###### How the Scenario Works

- Subscriptions: Agents subscribe to topics based on the tenant's identity.
- Publishing: Messages are published to the tenant-specific topic.
- Message Handling: Agents only receive messages relevant to their tenant.

###### Benefits
- Tenant Isolation: Ensures data privacy and separation between clients.
- Collaboration Within Tenant: Agents can collaborate freely within their tenant.

###### Considerations
- Complexity: Requires managing multiple sets of agents and topics.
- Resource Usage: More agent instances may consume additional resources.

```python
async def run_multi_tenant_single_scope() -> None:
    # Create the runtime
    runtime = SingleThreadedAgentRuntime()

    # List of clients (tenants)
    tenants = ["ClientABC", "ClientXYZ"]

    # Initialize sessions and map the topic type to each TaxSpecialist agent type
    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type,
            factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                description=f"A tax specialist in {specialty.value}.",
                specialty=specialty,
                system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
            ),
        )
        specialist_subscription = DefaultSubscription(agent_type=specialist_agent_type)
        await runtime.add_subscription(specialist_subscription)

    # Start the runtime
    runtime.start()

    # Publish client requests to their respective topics
    for tenant in tenants:
        topic_source = tenant  # The topic source is the client name
        topic_id = DefaultTopicId(source=topic_source)
        await runtime.publish_message(
            ClientRequest(f"{tenant} requires tax services."),
            topic_id=topic_id,
        )

    # Allow time for message processing
    await asyncio.sleep(1)

    # Stop the runtime when idle
    await runtime.stop_when_idle()


await run_multi_tenant_single_scope()
```

```text
==================================================
Tax specialist TaxSpecialist_planning:ClientABC with specialty TaxSpecialty.PLANNING:
ClientABC requires tax services.

==================================================
Tax specialist TaxSpecialist_dispute_resolution:ClientABC with specialty TaxSpecialty.DISPUTE_RESOLUTION:
ClientABC requires tax services.

==================================================
Tax specialist TaxSpecialist_compliance:ClientABC with specialty TaxSpecialty.COMPLIANCE:
ClientABC requires tax services.

==================================================
Tax specialist TaxSpecialist_preparation:ClientABC with specialty TaxSpecialty.PREPARATION:
ClientABC requires tax services.

==================================================
Tax specialist TaxSpecialist_planning:ClientXYZ with specialty TaxSpecialty.PLANNING:
ClientXYZ requires tax services.

==================================================
Tax specialist TaxSpecialist_dispute_resolution:ClientXYZ with specialty TaxSpecialty.DISPUTE_RESOLUTION:
ClientXYZ requires tax services.

==================================================
Tax specialist TaxSpecialist_compliance:ClientXYZ with specialty TaxSpecialty.COMPLIANCE:
ClientXYZ requires tax services.

==================================================
Tax specialist TaxSpecialist_preparation:ClientXYZ with specialty TaxSpecialty.PREPARATION:
ClientXYZ requires tax services.
```

##### 3. Single-Tenant, Multiple Scopes of Publishing

###### Scenario Explanation

In the single-tenant, multiple scopes of publishing scenario:

- All agents operate within a single tenant.
- Messages are published to different topics.
- Agents subscribe to specific topics relevant to their role or specialty.
- Messages are directed to subsets of agents based on the topic.

This scenario allows for targeted communication within a tenant, enabling more granular control over message distribution.

###### Application in the Tax Management Company

In this scenario:

- The tax system manager communicates with specific specialists based on their specialties.
- Different topics represent different specialties (e.g., "planning", "compliance").
- Specialists subscribe only to the topic that matches their specialty.
- The manager publishes messages to specific topics to reach the intended specialists.

###### How the Scenario Works

- Subscriptions: Agents subscribe to topics corresponding to their specialties.
- Publishing: Messages are published to topics based on the intended recipients.
- Message Handling: Only agents subscribed to a topic receive its messages.
###### Benefits

- Targeted Communication: Messages reach only the relevant agents.
- Efficiency: Reduces unnecessary message processing by agents.

###### Considerations

- Setup Complexity: Requires careful management of topics and subscriptions.
- Flexibility: Changes in communication scenarios may require updating subscriptions.

```python
async def run_single_tenant_multiple_scope() -> None:
    # Create the runtime
    runtime = SingleThreadedAgentRuntime()
    # Register TaxSpecialist agents for each specialty and add subscriptions
    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type,
            factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                description=f"A tax specialist in {specialty.value}.",
                specialty=specialty,
                system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
            ),
        )
        specialist_subscription = TypeSubscription(topic_type=specialty.value, agent_type=specialist_agent_type)
        await runtime.add_subscription(specialist_subscription)

    # Start the runtime
    runtime.start()

    # Publish a ClientRequest to each specialist's topic
    for specialty in TaxSpecialty:
        topic_id = TopicId(type=specialty.value, source="default")
        await runtime.publish_message(
            ClientRequest(f"I need assistance with {specialty.value} taxes."),
            topic_id=topic_id,
        )

    # Allow time for message processing
    await asyncio.sleep(1)

    # Stop the runtime when idle
    await runtime.stop_when_idle()


await run_single_tenant_multiple_scope()
```

```text
==================================================
Tax specialist TaxSpecialist_planning:default with specialty TaxSpecialty.PLANNING:
I need assistance with planning taxes.

==================================================
Tax specialist TaxSpecialist_dispute_resolution:default with specialty TaxSpecialty.DISPUTE_RESOLUTION:
I need assistance with dispute_resolution taxes.

==================================================
Tax specialist TaxSpecialist_compliance:default with specialty TaxSpecialty.COMPLIANCE:
I need assistance with compliance taxes.

==================================================
Tax specialist TaxSpecialist_preparation:default with specialty TaxSpecialty.PREPARATION:
I need assistance with preparation taxes.
```

##### 4. Multi-Tenant, Multiple Scopes of Publishing

###### Scenario Explanation

In the multi-tenant, multiple scopes of publishing scenario:

- There are multiple tenants, each with their own set of agents.
- Messages are published to multiple topics within each tenant.
- Agents subscribe to tenant-specific topics relevant to their role.
- Combines tenant isolation with targeted communication.

This scenario provides the highest level of control over message distribution, suitable for complex systems with multiple clients and specialized communication needs.

###### Application in the Tax Management Company

In this scenario:

- The company serves multiple clients, each with dedicated agent instances.
- Within each client, agents communicate using multiple topics based on specialties.
- For example, Client A's planning specialist subscribes to the "planning" topic with source "ClientA".
- The tax system manager for each client communicates with their specialists using tenant-specific topics.

###### How the Scenario Works

- Subscriptions: Agents subscribe to topics based on both tenant identity and specialty.
- Publishing: Messages are published to tenant-specific and specialty-specific topics.
- Message Handling: Only agents matching the tenant and topic receive messages.

###### Benefits

- Complete Isolation: Ensures both tenant and communication isolation.
- Granular Control: Enables precise routing of messages to intended agents.

###### Considerations

- Complexity: Requires careful management of topics, tenants, and subscriptions.
- Resource Usage: Increased number of agent instances and topics may impact resources.

```python
async def run_multi_tenant_multiple_scope() -> None:
    # Create the runtime
    runtime = SingleThreadedAgentRuntime()

    # Define TypeSubscriptions for each specialty and tenant
    tenants = ["ClientABC", "ClientXYZ"]

    # Initialize agents for all specialties and add type subscriptions
    for specialty in TaxSpecialty:
        specialist_agent_type = f"TaxSpecialist_{specialty.value}"
        await TaxSpecialist.register(
            runtime=runtime,
            type=specialist_agent_type,
            factory=lambda specialty=specialty: TaxSpecialist(  # type: ignore
                description=f"A tax specialist in {specialty.value}.",
                specialty=specialty,
                system_messages=[SystemMessage(content=f"You are a tax specialist in {specialty.value}.")],
            ),
        )
        for tenant in tenants:
            specialist_subscription = TypeSubscription(
                topic_type=f"{tenant}_{specialty.value}", agent_type=specialist_agent_type
            )
            await runtime.add_subscription(specialist_subscription)

    # Start the runtime
    runtime.start()

    # Send messages for each tenant to each specialty
    for tenant in tenants:
        for specialty in TaxSpecialty:
            topic_id = TopicId(type=f"{tenant}_{specialty.value}", source=tenant)
            await runtime.publish_message(
                ClientRequest(f"{tenant} needs assistance with {specialty.value} taxes."),
                topic_id=topic_id,
            )

    # Allow time for message processing
    await asyncio.sleep(1)

    # Stop the runtime when idle
    await runtime.stop_when_idle()


await run_multi_tenant_multiple_scope()
```

```text
==================================================
Tax specialist TaxSpecialist_planning:ClientABC with specialty TaxSpecialty.PLANNING:
ClientABC needs assistance with planning taxes.

==================================================
Tax specialist TaxSpecialist_dispute_resolution:ClientABC with specialty TaxSpecialty.DISPUTE_RESOLUTION:
ClientABC needs assistance with dispute_resolution taxes.

==================================================
Tax specialist TaxSpecialist_compliance:ClientABC with specialty TaxSpecialty.COMPLIANCE:
ClientABC needs assistance with compliance taxes.

==================================================
Tax specialist TaxSpecialist_preparation:ClientABC with specialty TaxSpecialty.PREPARATION:
ClientABC needs assistance with preparation taxes.

==================================================
Tax specialist TaxSpecialist_planning:ClientXYZ with specialty TaxSpecialty.PLANNING:
ClientXYZ needs assistance with planning taxes.

==================================================
Tax specialist TaxSpecialist_dispute_resolution:ClientXYZ with specialty TaxSpecialty.DISPUTE_RESOLUTION:
ClientXYZ needs assistance with dispute_resolution taxes.

==================================================
Tax specialist TaxSpecialist_compliance:ClientXYZ with specialty TaxSpecialty.COMPLIANCE:
ClientXYZ needs assistance with compliance taxes.

==================================================
Tax specialist TaxSpecialist_preparation:ClientXYZ with specialty TaxSpecialty.PREPARATION:
ClientXYZ needs assistance with preparation taxes.
```


## Structured output using GPT-4o models

*[cookbook/structured-output-agent.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/structured-output-agent.html)*

### Structured output using GPT-4o models

This cookbook demonstrates how to obtain structured output using GPT-4o models. The OpenAI beta client SDK provides a parse helper that allows you to use your own Pydantic model, eliminating the need to define a JSON schema. This approach is recommended for supported models.

Currently, this feature is supported for:

- gpt-4o-mini on OpenAI
- gpt-4o-2024-08-06 on OpenAI
- gpt-4o-2024-08-06 on Azure

Let's define a simple message type that carries explanation and output for a Math problem

```python
from pydantic import BaseModel


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str
```

```python
import os

# Set the environment variable
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://YOUR_ENDPOINT_DETAILS.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-2024-08-06"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"
```

```python
import json
import os
from typing import Optional

from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# Function to get environment variable and ensure it is not None
def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value


# Create the client with type-checked environment variables
client = AzureOpenAIChatCompletionClient(
    azure_deployment=get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model=get_env_variable("AZURE_OPENAI_MODEL"),
    api_version=get_env_variable("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=get_env_variable("AZURE_OPENAI_ENDPOINT"),
    api_key=get_env_variable("AZURE_OPENAI_API_KEY"),
)
```

```python
# Define the user message
messages = [
    UserMessage(content="What is 16 + 32?", source="user"),
]

# Call the create method on the client, passing the messages and additional arguments
# The extra_create_args dictionary includes the response format as MathReasoning model we defined above
# Providing the response format and pydantic model will use the new parse method from beta SDK
response = await client.create(messages=messages, extra_create_args={"response_format": MathReasoning})

# Ensure the response content is a valid JSON string before loading it
response_content: Optional[str] = response.content if isinstance(response.content, str) else None
if response_content is None:
    raise ValueError("Response content is not a valid JSON string")

# Print the response content after loading it as JSON
print(json.loads(response_content))

# Validate the response content with the MathReasoning model
MathReasoning.model_validate(json.loads(response_content))
```

```text
{'steps': [{'explanation': 'Start by aligning the numbers vertically.', 'output': '\n  16\n+ 32'}, {'explanation': 'Add the units digits: 6 + 2 = 8.', 'output': '\n  16\n+ 32\n   8'}, {'explanation': 'Add the tens digits: 1 + 3 = 4.', 'output': '\n  16\n+ 32\n  48'}], 'final_answer': '48'}
MathReasoning(steps=[Step(explanation='Start by aligning the numbers vertically.', output='\n  16\n+ 32'), Step(explanation='Add the units digits: 6 + 2 = 8.', output='\n  16\n+ 32\n   8'), Step(explanation='Add the tens digits: 1 + 3 = 4.', output='\n  16\n+ 32\n  48')], final_answer='48')
```


## Tracking LLM usage with a logger

*[cookbook/llm-usage-logger.ipynb](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/cookbook/llm-usage-logger.html)*

### Tracking LLM usage with a logger

The model clients included in AutoGen emit structured events that can be used to track the usage of the model. This notebook demonstrates how to use the logger to track the usage of the model.

These events are logged to the logger with the name: :py:attr:`autogen_core.EVENT_LOGGER_NAME`.

```python
import logging

from autogen_core.logging import LLMCallEvent


class LLMUsageTracker(logging.Handler):
    def __init__(self) -> None:
        """Logging handler that tracks the number of tokens used in the prompt and completion."""
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record. To be used by the logging module."""
        try:
            # Use the StructuredMessage if the message is an instance of it
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens
        except Exception:
            self.handleError(record)
```

Then, this logger can be attached like any other Python logger and the values read after the model is run.

```python
from autogen_core import EVENT_LOGGER_NAME

# Set up the logging configuration to use the custom handler
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.INFO)
llm_usage = LLMUsageTracker()
logger.handlers = [llm_usage]

# client.create(...)

print(llm_usage.prompt_tokens)
print(llm_usage.completion_tokens)
```


---

# FAQs


## FAQs

*[faqs.md](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/faqs.html)*

### FAQs

#### How do I get the underlying agent instance?

Agents might be distributed across multiple machines, so the underlying agent instance is intentionally discouraged from being accessed. If the agent is definitely running on the same machine, you can access the agent instance by calling {py:meth}`autogen_core.AgentRuntime.try_get_underlying_agent_instance` on the `AgentRuntime`. If the agent is not available this will throw an exception.

#### How do I call call a function on an agent?

Since the instance itself is not accessible, you can't call a function on an agent directly. Instead, you should create a type to represent the function call and its arguments, and then send that message to the agent. Then in the agent, create a handler for that message type and implement the required logic. This also supports returning a response to the caller.

This allows your agent to work in a distributed environment a well as a local one.

#### Why do I need to use a factory to register an agent?

An {py:class}`autogen_core.AgentId` is composed of a `type` and a `key`. The type corresponds to the factory that created the agent, and the key is a runtime, data dependent key for this instance.

The key can correspond to a user id, a session id, or could just be "default" if you don't need to differentiate between instances. Each unique key will create a new instance of the agent, based on the factory provided. This allows the system to automatically scale to different instances of the same agent, and to manage the lifecycle of each instance independently based on how you choose to handle keys in your application.

#### How do I increase the GRPC message size?

If you need to provide custom gRPC options, such as overriding the `max_send_message_length` and `max_receive_message_length`, you can define an `extra_grpc_config` variable and pass it to both the `GrpcWorkerAgentRuntimeHost` and `GrpcWorkerAgentRuntime` instances.

```python
# Define custom gRPC options
extra_grpc_config = [
    ("grpc.max_send_message_length", new_max_size),
    ("grpc.max_receive_message_length", new_max_size),
]

# Create instances of GrpcWorkerAgentRuntimeHost and GrpcWorkerAgentRuntime with the custom gRPC options

host = GrpcWorkerAgentRuntimeHost(address=host_address, extra_grpc_config=extra_grpc_config)
worker1 = GrpcWorkerAgentRuntime(host_address=host_address, extra_grpc_config=extra_grpc_config)
```

**Note**: When `GrpcWorkerAgentRuntime` creates a host connection for the clients, it uses `DEFAULT_GRPC_CONFIG` from `HostConnection` class as default set of values which will can be overriden if you pass parameters with the same name using `extra_grpc_config`.

#### What are model capabilities and how do I specify them?

Model capabilites are additional capabilities an LLM may have beyond the standard natural language features. There are currently 3 additional capabilities that can be specified within Autogen

- vision: The model is capable of processing and interpreting image data.
- function_calling: The model has the capacity to accept function descriptions; such as the function name, purpose, input parameters, etc; and can respond with an appropriate function to call including any necessary parameters.
- json_output: The model is capable of outputting responses to conform with a specified json format.

Model capabilities can be passed into a model, which will override the default definitions. These capabilities will not affect what the underlying model is actually capable of, but will allow or disallow behaviors associated with them. This is particularly useful when [using local LLMs](cookbook/local-llms-ollama-litellm.ipynb).

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="YourApiKey",
    model_capabilities={
        "vision": True,
        "function_calling": False,
        "json_output": False,
    }
)
```


