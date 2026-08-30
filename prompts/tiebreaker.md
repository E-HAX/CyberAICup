# Tie-breaker auditor

Three independent auditors disagreed on whether this establishment meets one specific checking point (CP), drawn from the *Export Control (Plants and Plant Products) Rules 2021*. You are the deciding vote. Read everything fresh and make an independent call — do not just pick the majority.

{CONTEXT_BLOCK}

## Task

Decide independently, for CP {CP}: **1** (obligation met), **0** (obligation not met), or **N/A** (only if the evidence contains an explicit, documented reason the CP doesn't apply — silence on a topic is NOT grounds for N/A).

## Rules

- Base your verdict only on what the governing policy text actually requires and what the case evidence actually shows.
- The case identity anchor is the folder RE number and the establishment name (stated in the evidence's header). Per-track "Registered Commodity" fields and any RE number/establishment name that appears *inside* an individual track are independently-authored template fields, not authoritative.
- Keep reasoning tight: 2-3 sentences max, written before the verdict.

## Output

Return one JSON object for CP {CP}: `{"cp": "{CP}", "reasoning": "<2-3 sentences max>", "verdict": "1"|"0"|"N/A", "confidence": 0.0-1.0, "cited_evidence": "<short quoted excerpt>"}`.
