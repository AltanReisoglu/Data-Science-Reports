One of the most expensive failure modes in AI agents isn't a crash. It's a silent retry loop.
An agent hits a rate limit, blindly retries until it succeeds, and the user never knows. No ticket gets filed. But under the hood, unnecessary API calls are burning tokens and adding latency that compounds at scale. Here's where that pattern shows up.
This stock analysis app runs two agents, a growth analyst building the bull case, and a risk analyst building the bear case, scoring a stock from opposing perspectives until they reach consensus.
Let's start with a healthy run. The app is asked to analyze Apple stock. The market data is fetched. The growth analyst scores the bull case. The risk analyst scores the bear case. They reach consensus, and a final score is returned.
Clean, fast rounds of analysis.
Now, here's the same request, but this time the tool call is flaky and runs into some errors when the app tries to fetch market data.
The user still gets their bull bear analysis and a consensus score. Same format, same confidence.
From the outside, nothing looks wrong, and that's exactly the problem.
This is the kind of failure that never gets a ticket filed against it. Silent, but deadly.
Hidden retries compounding into real cost and real latency. Let's go see what actually happened. Over in the Galileo console, the log stream tells a different story.
The first trace of the clean run shows no issues, no incomplete tool calls, no tool errors.
The second trace has immediate red flags. Tool error rate has spiked.
But the analysts still reached a consensus. Clicking into the trace view of the second call, the problem is immediately visible. Everything looks normal except the parent span, with two failed tool calls before a success. Two
retries might seem minor, but it's the pattern that matters. This retry logic has no circuit breaker and no timeout guard.
A transient error today could be a sustained outage tomorrow with every request burning through max retries and blocking the entire downstream pipeline.
Without writing a single line of debugging code, the what and the where are both clear. Fragile retry logic in the get stock info tool call. Not the analysts, not the consensus logic. The agents performed correctly. The vulnerability is in the data fetching layer.
What used to be hours of log diving, Galileo collapses into seconds.
Right down to the span.
How would your team detect this pattern today?
