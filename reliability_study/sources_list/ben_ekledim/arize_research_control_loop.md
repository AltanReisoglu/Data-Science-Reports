What Is An Agent Control Loop?

Agent control loop
An agent control loop is the repeated cycle of observe state, decide the next action, execute that action, update state, and continue until a stopping condition is met. It is the runtime loop that turns a model into an agent. A single model call maps one input to one output. A control loop takes that output, does something with it, and feeds the result back in.

Most of what is hard about agents lives here. The model is stateless between calls, so the loop carries progress forward, and the loop decides when to stop. A loop with no credible stopping condition is the difference between an agent that finishes a task and one that burns its entire budget discovering a file does not exist.

Key takeaways
The loop has four phases: observe state, decide an action, execute it, write the result back. The model only handles the decide phase.
Every loop needs a hard stop that does not depend on the model’s judgment, since the model is the component most likely to be wrong about whether it is done.
Step limits, cumulative token or dollar budgets, and wall-clock deadlines catch most runaway behavior. Loop detection catches the rest.
The common non-terminating failure is repetition: the same tool called with the same arguments, producing the same unhelpful result.
Record a stop reason on every run. completed, max_steps, budget_exceeded, and error are different outcomes, and averaging them hides what went wrong.
The four phases
Observe. Assemble what the model sees this iteration: the task, the message history or a summary of it, the last tool result, and what the state record says has already happened. This is context assembly, and most quality problems start here. The model cannot act on information the observe phase did not give it.

Decide. Call the model and parse the output into an action, usually a tool call with structured arguments or a signal that the task is finished. Parse failures belong here and are more common than they should be. An unparseable action is a decision the loop cannot execute, so it needs a defined response: retry once with the parse error appended, then fail the step rather than retrying forever.

Act. Execute against the real world, under tool permissions, timeouts, and retry policy. This is the only phase with side effects, which matters as soon as you think about resuming a run after a crash.

Update. Write the result into state and increment the counters the loop reads to decide whether to continue.

The agent harness is the code that owns all four phases. Swapping the model changes the decide phase. Changing the retry policy, the step limit, or how much history gets assembled changes behavior just as much, which is why loop configuration gets versioned alongside the prompt.

What actually stops the loop
Five stopping conditions cover almost every production agent:

Task completion. The model signals it is done, usually by calling a terminal tool instead of another action. The intended exit, and the least reliable one.
Step limit. A maximum number of iterations. The single most effective guardrail, because it does not depend on anything the model says.
Budget cap. Cumulative tokens or dollars, which catches the run whose step count looks fine but whose context grew until each step got expensive.
Deadline. Wall-clock time, which matters most when tools are slow or the agent is waiting on an external system.
Error policy. Consecutive tool failures, a permission denial, or a policy check that fails hard.
Pick the limits from measured runs, not from a number someone liked. Look at the step count distribution for tasks that succeeded and set the ceiling above its tail. A step limit tight enough to cut off legitimate work becomes a silent quality regression that looks like the model got worse.

When nothing stops it
Runaway loops are usually repetitive, not creative. Three patterns show up repeatedly:

Identical retries. The agent calls the same tool with the same arguments, gets the same error, and tries again because nothing in the assembled context tells it this already failed. Hash the tool name plus normalized arguments, keep recent hashes in state, and when one repeats, inject the prior result explicitly or block the call.

Oscillation. The agent alternates between two actions that undo each other, editing a file back and forth being the canonical version. A no-progress detector helps: if state has not meaningfully changed across several iterations, stop and escalate.

Plan thrash. The agent re-plans without executing, usually because the plan is in the context but the evidence of prior attempts is not. That is a context assembly bug wearing a reasoning costume.

Detecting these requires per-iteration telemetry. If the whole run is one span, you cannot see that steps 4 through 19 were the same call. A span per iteration, tagged with the iteration index, tool name, and argument hash, makes the repetition obvious, and that per-step view is the baseline for tracing and evaluating agents rather than scoring only the final answer.

Inner loop and outer loop
The inner loop is the tool-use cycle inside one task. The outer loop is the cycle of running the agent, reviewing failures, and changing prompts or tools, which is the agent lifecycle. Security controls attach to the inner one, because each iteration is a fresh chance for injected content in a tool result to redirect the next decision. Permission checks belong on the act phase of every iteration, and where they sit relative to planning and tool selection is part of the agent architecture you choose.

FAQ
How do iterative loops work in modern agent architectures?
The structure is the same across architectures: a model proposes an action, the runtime executes it, the result is added to what the model sees next, and the cycle repeats. What differs is who decides. Some designs let the model choose freely from a tool list at every step, which handles tasks you could not enumerate in advance. Others constrain it to a fixed sequence of stages with a model call inside each, which is easier to evaluate and harder to break.

What is the difference between a control loop and a workflow?
A workflow has its control flow written by you: step one, then step two, with branches you defined. A control loop delegates the next-step decision to the model at runtime. Most production systems are a mix, with a deterministic workflow calling a bounded agentic loop for the part of the task that needs it.

How many steps should an agent be allowed to take?
There is no correct number, and any figure quoted without context is a guess. Derive it from your own runs: measure the step counts of runs that produced good outcomes and set the ceiling above that range. Then track how often runs terminate on the limit. A rate that climbs usually means task difficulty or tool reliability changed, not that the limit is wrong.

How do you keep the loop from taking unsafe actions?
Treat every iteration as untrusted. Check permissions at execution time rather than granting them for the whole run, keep destructive tools behind an explicit approval step, and validate tool arguments against a schema before the call. Content returned by a tool is input, not instruction, and the loop should be built on that assumption.

Where does evaluation fit?
Inline checks run inside the loop and can change behavior, such as rejecting a malformed action before it executes, so they have to be fast. Trajectory evaluation runs on the recorded run afterward and can afford a judge model, because nothing is waiting on it.