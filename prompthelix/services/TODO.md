# Services TODO

Items for service layer expansion.

- [x] Replace in-memory PromptManager with database-backed `PromptService`
  (the old `PromptManager` class remains for backward compatibility but is
  deprecated)
- [ ] Add caching layer using Redis
- [ ] Implement background workers for long-running tasks
- [ ] Provide service interfaces for agent coordination
