# Example procedure: greeting flow

This is a sample procedural-memory file. Wild Memory loads every `.md`
file in `procedures_dir` and exposes them to the agent as named workflows.

## Trigger

Trigger this procedure when the conversation starts and the agent has
no prior context for the user.

## Steps

1. **Greet the user briefly.** Avoid scripts — the imprint defines tone.
2. **Ask a single open-ended question** to learn the user's intent.
3. **Listen, save.** When the user shares a fact, the Bee distiller will
   capture it as an observation. Don't paraphrase facts back unless the
   user asks for confirmation.
4. **Hand off** if the user requests a human or asks something out of
   scope. Record a `handoff_request` feedback signal.

## Notes

- One question per turn. Multi-question messages overwhelm.
- Procedure success is tracked per step in `feedback_signals` and feeds
  back into the Chameleon layer.
