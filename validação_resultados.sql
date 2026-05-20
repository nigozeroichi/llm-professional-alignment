WITH all_results AS (
    SELECT *, 'Crim' AS topico FROM './Crim_Experiment_Results.csv'
    UNION ALL
    SELECT *, 'Freexp' AS topico FROM './Freexp_Experiment_Results.csv'
    UNION ALL
    SELECT *, 'Ineq' AS topico FROM './Ineq_Experiment_Results.csv'
    UNION ALL
    SELECT *, 'Vdem' AS topico FROM './Vdem_Experiment_Results.csv'
)
SELECT
    topico,
    llm_choice,
    COUNT(*) AS quantidade
FROM all_results
GROUP BY topico, llm_choice
ORDER BY topico, quantidade DESC;