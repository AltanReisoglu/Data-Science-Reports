[sdeoffer]
System Design
Tech Stacks
AI Coding
DDIA
Leetcode
Search…
EN
中
Mock Interview →
home
/
ai-coding
/
loop-engineering
AI Coding
Loop Engineering — Designing the Agent Loop
Jun 21, 2026
·
13 min read
·
AI Coding
A model call is one shot: text in, text out, done. An agent is what happens when you wrap that call in a loop — call the model, run what it asked for, feed the result back, and call it again — until the job is finished. That loop is where an agent does all of its exploring, acting, and self-correcting, and designing it well is its own discipline: loop engineering. As frontier models converge in raw ability and "context engineering" became the buzzword for what you feed the model, loop engineering is the next layer down — the control flow that decides when to think, when to act, when to stop, and what to do when things go wrong. This is the companion to harness engineering: the harness is the whole car; this piece is about the engine's combustion cycle.

⚡ Quick Takeaways
The loop is what turns a one-shot model into an agent. Call → act → observe → repeat is the cycle where exploration, action, and self-correction happen.
Stopping is the hardest part. Knowing when the task is actually done — and not stopping early or looping forever — is the central control-flow problem.
Every loop needs budgets. Hard caps on steps, tokens, time, and cost are non-negotiable: they're the backstop against a confused agent burning unbounded resources.
Verification closes the loop. Running tests/builds and feeding failures back is what makes the loop converge on a correct answer rather than just spin.
Recovery is most of the engineering. Tool errors, malformed output, and the agent repeating a failing action ("doom loops") need retries, loop detection, and escape hatches.
Topology is a design choice: single loop, reflection (act → critique → retry), plan-then-execute, and orchestrator + sub-agents each fit different tasks.
You tune the loop empirically — small changes to stop conditions, budgets, or context handling swing real success rates, so you measure with evals and trajectory analysis.
tldr
Loop engineering is the discipline of designing the agent's control loop: the plan-act-observe cycle, the stopping conditions that decide when it's done, the step/token/time/cost budgets that bound it, the verification that grounds it in the environment, the recovery logic that handles errors and breaks doom loops, and the topology (single, reflection, plan-execute, sub-agents) that fits the task. Because models are commoditizing, how well you engineer this loop is increasingly what separates an agent that finishes the job from one that flails.

THE AGENT LOOP — one iteration
LOOP CONTROLLER
EXIT
Assemble context
system + history + obs
Call model
the only LLM call
Decide
tool call? or done?
Execute action
tool / cmd
Observe result
append to state
Budget: steps · tokens · cost
Stop conditions
Verifier (tests / build)
Recovery · loop detect
Final answer
verified · within budget
Give up / escalate
act
loop back
done
governs
Anatomy of the agent loop — assemble context → call model → decide → execute → observe → loop back, with a controller (budgets, stop conditions, verifier, recovery) governing every iteration and an exit only when the answer is verified or the budget is spent
What the Loop Actually Is
Strip away the framework names and every agent is the same five lines: maintain a running state (the conversation, plus everything observed so far), call the model on it, and if the model asks to use a tool, execute that tool, append the result to the state, and call the model again. Repeat until the model says it's done or you hit a limit. That's it — the loop is small. What makes it hard is everything you have to decide around those five lines.

the loop — and the decisions hiding in it
copy
state = [system, tools, goal]
steps, spent = 0, 0
while not done(state) and steps < MAX_STEPS and spent < BUDGET:   # ← stop + budget
    reply = model(state)                          # the only LLM call
    if reply.tool_calls:
        for call in reply.tool_calls:
            obs = execute(call)                 # ← errors? timeouts? recovery
            state.append(obs)
    else:
        if verify(reply): return reply        # ← is it actually correct?
        state.append("verification failed: …")   # ← feed failure back, keep going
    state = compact(state)                     # ← context across iterations
    steps, spent = steps + 1, spent + reply.cost
return give_up(state)                          # ← graceful failure / escalate
Each comment marks a loop-engineering decision: when is done true, how big are the budgets, how do you recover from a failed execute, what does verify check, how do you compact state so it fits the window, and what happens when you fall out the bottom. The rest of this article is those decisions.

Why the Loop Is Now the Product
For a few years the story was "better model wins." That's flattening: frontier models are converging, and the same model dropped into two different loops can have wildly different success rates on real tasks. The loop controls what the model sees each turn, how many attempts it gets, whether it catches its own mistakes, and whether it quits gracefully or burns your budget. So the differentiator has moved up the stack: first to context engineering (what you feed the model), and now to loop engineering (how you orchestrate repeated calls into reliable work). Coding agents that "just feel more capable" usually have a better-engineered loop, not a secret model.

The Control Flow: Plan, Act, Observe
The canonical loop body is a three-beat cycle, often called ReAct (reason + act): the model reasons about what to do next, emits an action (a tool call), and the harness returns an observation that goes back into context for the next reason step. The power is in the feedback: the agent isn't predicting a whole solution up front, it's taking one grounded step at a time, adapting to what each action reveals.

The first real design choice lives here: how much to do per turn. A single tool call per step is simplest and easiest to recover from, but slow and chatty. Allowing the model to request several tool calls at once (parallel reads, say) cuts latency and token cost — at the price of harder error handling when one of them fails. Most production loops allow batched read-only calls but serialize anything that mutates state, so a failure is localized.

Stopping Conditions — the Hard Part
The deceptively hard question in any loop is "are we done?" Stop too early and the agent hands back a half-finished task; never stop and it loops forever, or keeps "improving" a fine answer until it breaks it. There is no single right answer — you compose several signals:

Model-declared completion — the model stops calling tools and emits a final answer. Necessary but not sufficient: models declare victory prematurely all the time.
Verification-gated — "done" only counts if a check passes (tests green, build succeeds, output matches a schema). This is the strongest signal because it's grounded in the environment, not the model's opinion.
Goal/exit criteria — an explicit predicate ("the function exists and all tests pass") the loop evaluates, rather than trusting prose.
Budget exhaustion — a hard stop regardless of state; the backstop, not the goal.
key point
Never let the model be the sole judge of "done." A model declaring success is a request to stop, not proof of completion — gate it on a verification check whenever the task has one. "It said it was finished" is the agent equivalent of "it compiles on my machine."

Budgets: Steps, Tokens, Time, Cost
An LLM in a loop is an unbounded spender by default — a confused agent will happily take 200 steps and burn dollars going nowhere. Budgets are the hard limits that make a loop safe to run unattended, and a serious loop tracks several at once:

Budget	What it caps / why
Step / iteration count	The simplest backstop — bounds how many model calls a single task can make, so a stuck agent halts instead of spinning.
Token budget	Bounds total context + generation tokens; the truest proxy for cost, since spend scales with tokens, not turns.
Wall-clock / timeout	Caps a single tool call (a hung command) and the whole task; protects latency-sensitive callers.
Dollar cost	A per-task or per-user ceiling; what you actually bill against and the limit ops cares about.
The subtlety isn't having a budget, it's what to do as you approach it. A good loop degrades gracefully: as the step budget runs low it can switch from exploring to wrapping up ("you have 2 steps left — summarize what you found and stop"), rather than getting cut off mid-thought. Budgets aren't just a kill switch; they're a signal the loop can reason about.

Verification: Closing the Loop
A loop without verification just iterates; a loop with it converges. The single feature that most separates a real agent from a fancy autocomplete is feeding the result of running the work back into the loop: run the tests, the linter, the build, the query — and turn a failure into an observation the model reasons about and fixes. This grounds the agent in the environment instead of its own (often wrong) belief that the code is correct.

Verification is what makes the loop's iteration meaningful: each pass isn't a re-roll of the dice, it's a step toward a checked result. The quality of this depends on loop-engineering choices — what to run, how to surface a 5,000-line failure log as the few lines that matter, and when a partial pass is "good enough." A weak model with a tight verify-and-retry loop often beats a strong model with none. (Evaluating whether the loop actually converges is its own discipline — see evals for LLM apps.)

Error Recovery and Doom Loops
Agents fail in ways a single call never does, and most of the engineering in a production loop is the recovery paths. The big ones:

Tool errors — a command fails or a file is missing. The fix is to return a readable, actionable error as the observation ("file not found: did you mean X?"), not a raw stack trace — the error text is feedback the model acts on.
Malformed output — the model emits invalid JSON for a tool call. Re-prompt with the parse error, optionally with constrained decoding, rather than crashing the loop.
The doom loop — the agent repeats the same failing action over and over, convinced it'll work this time. This is the signature agent failure, and it needs explicit loop detection.
Loop detection means noticing repetition — the same tool call with the same args, or the same error N times — and breaking the pattern: inject a nudge ("that approach has failed twice; try something different"), escalate to a stronger model, or stop. Without it, a budget is your only protection, and you waste the whole budget on the same mistake. Pairing the two — detect the repeat early, fall back on the budget — is the standard belt-and-suspenders.

detecting a doom loop and breaking it
copy
recent = []
def guard(action, result):
    sig = (action.name, action.args, result.error)   # fingerprint the attempt
    recent.append(sig)
    if recent[-3:].count(sig) >= 3:               # same failing action 3×
        return "STUCK: this has failed repeatedly — change strategy or stop"
    return None                                  # otherwise continue normally
Loop Topologies
The plain while-loop is the baseline, but "loop engineering" increasingly means choosing the right shape of loop for the task. Four common topologies:

Single loop
act → observe
repeat until done
Reflection
draft
critique
revise
Plan → execute
make plan
s1
s2
s3
steps, each its own mini-loop
Orchestrator + sub-agents
main
sub A
sub B
sub C
fan out · merge results
Four loop topologies — a single act/observe loop; a reflection loop (draft → critique → revise); plan-then-execute (one plan, each step its own mini-loop); and an orchestrator that fans work out to sub-agents with clean contexts, then merges
Single loop — the default ReAct cycle. Best for open-ended tasks where the next step depends on the last observation; simplest to build and debug.
Reflection — after producing an answer, a separate critique step looks for flaws and triggers a revision. Adds a self-review beat that catches errors the first pass missed, at the cost of extra model calls.
Plan-then-execute — the agent drafts a plan first, then executes the steps (each potentially its own small loop). Keeps long tasks coherent and gives you a checkpoint to inspect the plan before spending on execution.
Orchestrator + sub-agents — a main loop delegates scoped jobs (search the codebase, review a diff) to fresh sub-agents with their own clean context, which return just a conclusion. This bounds how much any one context window must hold and enables parallelism — the topology behind most large-task agents.
Context Across Iterations
Every loop iteration appends observations to the state, so a long task inevitably bloats toward the context-window ceiling. Managing that growth is a core loop-engineering job — the loop has to keep the goal and the key facts in view while shedding noise. The standard move is compaction: when the state approaches the window, summarize or drop older turns, preserving the goal, the plan, and recent results. Related tactics include offloading detail to an external scratchpad the agent can re-read on demand, and the sub-agent pattern above, which keeps the main loop's context lean by sending heavy work to a child. Get this wrong and the agent "forgets" what it was doing mid-task; get it right and it can work on problems far longer than any single window.

Tuning the Loop Empirically
You cannot tune a loop by intuition, because the changes that matter — a different stop condition, a tighter budget, when to compact, whether to add a reflection beat — interact in ways that swing real-world success rates surprisingly far. Serious loop work is empirical: run the agent against benchmark tasks, measure success rate, steps, and cost, and inspect trajectories (the full sequence of calls and observations) to see where runs go wrong — premature stops, doom loops, budget blowouts. Then change one thing and re-measure. The loop is a system you tune against data, the same way you'd tune any control system.

interview tip
If asked to "design an agent," resist jumping to the model. Lead with the loop: "It's a call-act-observe cycle; the interesting decisions are the stopping condition, the budgets, in-loop verification, and recovery from doom loops." That framing shows you understand where agent reliability actually comes from.

Pitfalls and Tradeoffs
Trusting model-declared "done." Without a verification gate, the agent stops on its own say-so and ships half-finished work. Gate on a real check.
No loop detection. Relying on the budget alone means a doom loop wastes the entire budget on one mistake; detect repetition early.
Budgets with no graceful degradation. A hard cut-off mid-thought wastes the work done so far; let the loop wind down when it's near the limit.
Over-engineering the topology. Reflection and multi-agent add model calls, latency, and cost; many tasks are better served by a tight single loop. Add structure only when a simpler loop demonstrably fails.
Context rot. Skipping compaction lets the window fill with stale observations until quality silently degrades; manage state every iteration.
Tuning blind. Without trajectory analysis you fix the wrong thing — you see a low success rate but not whether it's early stops, loops, or bad recovery.
takeaway
An agent is a loop, and loop engineering is the discipline of designing it well. The model supplies the reasoning; the loop supplies the control flow that makes that reasoning into reliable work — deciding when to act, when to stop, how much to spend, how to verify, and how to recover. As models commoditize, this control flow is increasingly the product: the same model in a well-engineered loop finishes the job, and in a careless one flails until the budget runs out.

🎯 interview hot-takes
What is "the loop"? The control cycle that wraps an LLM call — call → act → observe → repeat — turning a one-shot model into an agent that explores and self-corrects.
What's the hardest part of a loop? The stopping condition: knowing when the task is truly done without stopping early or looping forever; gate it on verification, not the model's word.
Why do agents need budgets? An LLM in a loop spends unboundedly; hard caps on steps/tokens/time/cost are the backstop, ideally with graceful wind-down near the limit.
What's a doom loop and how do you stop it? The agent repeating a failing action; detect repetition (same call/error N times) and break the pattern — nudge, escalate, or stop — backed by the budget.
What makes the loop converge instead of spin? In-loop verification — running tests/build and feeding failures back grounds each iteration in the environment.
When would you use sub-agents? When a task is large or its context gets cluttered — delegate scoped work to fresh sub-agents with clean contexts and merge results, enabling parallelism.

