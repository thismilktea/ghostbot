The previous plan failed the planning quality check. Rewrite it into an approvable plan.

Rules:
- Use only read-only evidence already gathered unless more exploration is needed.
- Preserve the user's original request and revision feedback.
- Fix every listed quality failure, including missing or weak user intent, acceptance criteria, and non-goals.
- Do not propose implementation beyond the requested scope.
- Collapse overly detailed plans into 3-5 major stages and usually 10-20 user-visible checklist items.
- Merge tiny checklist items and combine related file edits into one stage.
- Preserve key constraints and acceptance checks, but do not expand a plan just because many files may change.
- Move critical boundaries into Requirements / Acceptance Criteria / Non-goals instead of repeating them as micro-steps.

Execution mode: {{ execution_mode }}
Required exploration steps: {{ min_exploration_steps }}
Tools used: {{ tools_used | join(', ') if tools_used else 'none' }}

Original request:
{{ request }}

{% if revision_feedback %}
Revision feedback:
{{ revision_feedback }}
{% endif %}

Weak plan:
{{ previous_plan }}

Quality failures:
{% for failure in quality_failures %}
- {{ failure }}
{% endfor %}

Rewrite instructions:
{% for instruction in rewrite_instructions %}
- {{ instruction }}
{% endfor %}

Return the rewritten plan using exactly these markdown sections:

## Execution Mode
## Summary
## User Intent
## Requirements
## Acceptance Criteria
## Non-goals / Out of Scope
## Exploration Evidence
## Proposed Approach
## Files Likely to Change
## Executable Checklist
## Verification Plan
## Risks and Open Questions
