-- Reconciliation views for analytics
CREATE OR REPLACE VIEW reconciliation.v_issue_summary AS
SELECT
    issue_type,
    resolution_status,
    COUNT(*) AS issue_count,
    SUM(ABS(COALESCE(amount_difference, 0))) AS total_amount_impact
FROM reconciliation.reconciliation_issues
GROUP BY issue_type, resolution_status;

CREATE OR REPLACE VIEW reconciliation.v_batch_health AS
SELECT
    b.batch_id,
    b.rows_ingested,
    b.rows_valid,
    b.rows_rejected,
    s.matched,
    s.missing_settlement,
    s.amount_mismatch,
    s.precision_score,
    s.recall_score,
    b.status,
    b.completed_at
FROM metadata.pipeline_batches b
LEFT JOIN reconciliation.summary s USING (batch_id);
