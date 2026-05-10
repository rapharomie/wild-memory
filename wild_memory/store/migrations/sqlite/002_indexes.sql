-- Indexes for the SQLite backend. Kept conservative; SQLite's planner
-- usually does fine without exhaustive indexing on small datasets.

CREATE INDEX IF NOT EXISTS idx_obs_agent_user_status
    ON observations (agent_id, user_id, status);

CREATE INDEX IF NOT EXISTS idx_obs_decay_status
    ON observations (status, decay_score);

CREATE INDEX IF NOT EXISTS idx_obs_obs_type
    ON observations (obs_type);

CREATE INDEX IF NOT EXISTS idx_refl_agent_user
    ON reflections (agent_id, user_id);

CREATE INDEX IF NOT EXISTS idx_feedback_session
    ON feedback_signals (session_id);

CREATE INDEX IF NOT EXISTS idx_feedback_agent_user
    ON feedback_signals (agent_id, user_id);

CREATE INDEX IF NOT EXISTS idx_citation_session
    ON citation_trails (session_id);

CREATE INDEX IF NOT EXISTS idx_proc_agent_status
    ON procedures (agent_id, status);

CREATE INDEX IF NOT EXISTS idx_cache_agent
    ON semantic_cache (agent_id);

CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON semantic_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_session_expires
    ON session_logs (expires_at);
