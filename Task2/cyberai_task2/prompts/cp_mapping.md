# OKF grounding pass — map checking points to governing policy sections

You are building the Open Knowledge Format (OKF) bundle for a compliance-audit system. For each official checking point (CP) below, identify exactly which section(s) of the *Export Control (Plants and Plant Products) Rules 2021* actually govern that obligation.

{POLICY_SECTIONS_BLOCK}

## Checking points to map

{CP_LIST_BLOCK}

## Task

For each CP above:
1. Read the CP's official text carefully.
2. Search the policy sections above for the section(s) that actually state this obligation — not sections that are merely topically adjacent.
3. Cite only sections you have actually read and confirmed state the obligation. Do not invent a citation, and do not guess based on a section's title alone.
4. If no section is a clean, direct match (e.g. the obligation is implied by a combination of sections, or the closest analogous section is in a different part of the Rules than you'd expect), say so explicitly in your rationale and cite the genuinely closest basis rather than forcing a citation to something that doesn't actually support it.

## Rules (no-hard-coding constraint)

- This mapping identifies WHICH TEXT governs a CP. It must NOT state or imply what evidence would satisfy or fail the CP — that determination happens later, separately, by reading actual case evidence against the cited text. Do not write anything resembling "this CP is met if X" or "look for Y in the evidence."
- Citations must be section identifiers from the list above (format: `policy/section_<id>.md`), exactly as given.

## Output

Return a JSON array, one entry per CP: `{"cp": "<CPn>", "citations": ["policy/section_X-Y.md", ...], "rationale": "<1-3 sentences: why these sections, and any judgment call made>"}`.
