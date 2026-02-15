"""
Reward calculator for Text-to-SQL RL training.
Implements execution-based and partial rewards.
"""

import difflib
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import timeout_decorator


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""

    execution_weight: float = 1.0
    partial_weight: float = 0.3
    timeout_seconds: int = 5
    use_partial_rewards: bool = True


class SQLRewardCalculator:
    """Calculate rewards for generated SQL queries."""

    def __init__(self, db_path: str, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator.

        Args:
            db_path: Path to SQLite database
            config: Reward configuration
        """
        self.db_path = db_path
        self.config = config or RewardConfig()

    def _execute_sql(
        self, sql: str, db_path: str
    ) -> Tuple[bool, Optional[List], Optional[str]]:
        """
        Execute SQL query with timeout and error handling.

        Returns:
            (success, results, error_message)
        """
        try:

            @timeout_decorator.timeout(self.config.timeout_seconds)
            def _run_query():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(sql)
                results = cursor.fetchall()
                conn.close()
                return results

            results = _run_query()
            return True, results, None

        except timeout_decorator.TimeoutError:
            return False, None, "Timeout"
        except sqlite3.Error as e:
            return False, None, f"SQL Error: {str(e)}"
        except Exception as e:
            return False, None, f"Execution Error: {str(e)}"

    def _normalize_results(self, results: List) -> set:
        """Normalize query results for comparison."""
        if results is None:
            return set()

        # Convert to set of tuples for order-independent comparison
        normalized = set()
        for row in results:
            # Convert all values to strings and lowercase for comparison
            normalized_row = tuple(
                str(val).lower() if val is not None else "null" for val in row
            )
            normalized.add(normalized_row)
        return normalized

    def execution_accuracy(self, pred_sql: str, gold_sql: str, db_path: str) -> float:
        """
        Compute execution-based reward.

        Returns:
            1.0 if results match, 0.0 otherwise
        """
        # Execute predicted SQL
        pred_success, pred_results, pred_error = self._execute_sql(pred_sql, db_path)

        if not pred_success:
            return 0.0

        # Execute gold SQL
        gold_success, gold_results, gold_error = self._execute_sql(gold_sql, db_path)

        if not gold_success:
            # Gold query should always work; if not, give partial credit
            return 0.0

        # Compare normalized results
        pred_normalized = self._normalize_results(pred_results)
        gold_normalized = self._normalize_results(gold_results)

        return 1.0 if pred_normalized == gold_normalized else 0.0

    def _extract_sql_components(self, sql: str) -> Dict[str, set]:
        """Extract SQL components for partial matching."""
        sql = sql.upper()

        components = {
            "select": set(),
            "from": set(),
            "where": set(),
            "group_by": set(),
            "order_by": set(),
            "keywords": set(),
        }

        # Extract SELECT columns
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.DOTALL)
        if select_match:
            components["select"] = set(select_match.group(1).split(","))

        # Extract FROM tables
        from_match = re.search(
            r"FROM\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)", sql
        )
        if from_match:
            components["from"] = set(from_match.group(1).split(","))

        # Extract WHERE conditions
        where_match = re.search(r"WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)", sql)
        if where_match:
            components["where"] = {where_match.group(1).strip()}

        # SQL keywords
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "LIMIT",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
        ]
        components["keywords"] = {kw for kw in keywords if kw in sql}

        return components

    def partial_rewards(self, pred_sql: str, gold_sql: str) -> float:
        """
        Compute partial rewards based on SQL structure similarity.

        Returns:
            Score in [0, 1] based on component overlap
        """
        pred_components = self._extract_sql_components(pred_sql)
        gold_components = self._extract_sql_components(gold_sql)

        scores = []

        # Compare each component type
        for comp_type in ["select", "from", "where", "keywords"]:
            pred_set = pred_components[comp_type]
            gold_set = gold_components[comp_type]

            if not gold_set:
                continue

            # Jaccard similarity
            intersection = len(pred_set & gold_set)
            union = len(pred_set | gold_set)

            if union > 0:
                scores.append(intersection / union)

        # Token-level similarity as fallback
        pred_tokens = set(pred_sql.upper().split())
        gold_tokens = set(gold_sql.upper().split())
        token_similarity = (
            len(pred_tokens & gold_tokens) / len(gold_tokens | pred_tokens)
            if gold_tokens
            else 0
        )
        scores.append(token_similarity)

        return sum(scores) / len(scores) if scores else 0.0

    def compute_reward(
        self, pred_sql: str, gold_sql: str, question: str, db_path: str
    ) -> Dict[str, float]:
        """
        Compute total reward combining execution and partial rewards.

        Args:
            pred_sql: Generated SQL query
            gold_sql: Ground truth SQL query
            question: Natural language question (unused, for future extensions)
            db_path: Path to database file

        Returns:
            Dictionary with reward components and total
        """
        # Execution-based reward
        # Attempt to execute and capture errors for debugging
        try:
            pred_success, pred_results, pred_error = self._execute_sql(
                pred_sql, db_path
            )
        except Exception as e:
            pred_success, pred_results, pred_error = (
                False,
                None,
                f"Execution exception: {e}",
            )

        # If execution succeeded, compute normalized equality
        if pred_success:
            # Execute gold SQL
            gold_success, gold_results, gold_error = self._execute_sql(
                gold_sql, db_path
            )
            if not gold_success:
                exec_reward = 0.0
            else:
                pred_normalized = self._normalize_results(pred_results)
                gold_normalized = self._normalize_results(gold_results)
                exec_reward = 1.0 if pred_normalized == gold_normalized else 0.0
        else:
            exec_reward = 0.0
            gold_success, gold_results, gold_error = self._execute_sql(
                gold_sql, db_path
            )

        # Partial rewards (if execution fails)
        if exec_reward == 0.0 and self.config.use_partial_rewards:
            partial_reward = self.partial_rewards(pred_sql, gold_sql)
        else:
            partial_reward = 0.0

        # Combined reward
        total_reward = (
            self.config.execution_weight * exec_reward
            + self.config.partial_weight * partial_reward
        )

        # Include debugging info about execution errors
        debug = {
            "pred_success": pred_success if "pred_success" in locals() else False,
            "pred_error": pred_error if "pred_error" in locals() else None,
            "gold_success": gold_success if "gold_success" in locals() else False,
            "gold_error": gold_error if "gold_error" in locals() else None,
        }

        return {
            "execution": exec_reward,
            "partial": partial_reward,
            "total": total_reward,
            "debug": debug,
        }

    def compute_batch_rewards(
        self,
        pred_sqls: List[str],
        gold_sqls: List[str],
        questions: List[str],
        db_paths: List[str],
    ) -> List[float]:
        """
        Compute rewards for a batch of predictions.

        Returns:
            List of total rewards
        """
        rewards = []

        for pred, gold, question, db_path in zip(
            pred_sqls, gold_sqls, questions, db_paths
        ):
            reward_dict = self.compute_reward(pred, gold, question, db_path)
            rewards.append(reward_dict["total"])

        return rewards


# Example usage and testing
if __name__ == "__main__":
    # Example test
    config = RewardConfig(
        execution_weight=1.0, partial_weight=0.3, use_partial_rewards=True
    )

    # This would need an actual database
    # reward_calc = SQLRewardCalculator(db_path="path/to/db.sqlite", config=config)

    # Test partial rewards without DB
    reward_calc = SQLRewardCalculator(db_path="dummy.db", config=config)

    pred_sql = "SELECT name, age FROM users WHERE age > 18 ORDER BY age"
    gold_sql = "SELECT name, age FROM users WHERE age > 18 ORDER BY age DESC"

    partial = reward_calc.partial_rewards(pred_sql, gold_sql)
    print(f"Partial reward: {partial:.3f}")

    # Test component extraction
    components = reward_calc._extract_sql_components(pred_sql)
    print(f"Components: {components}")
