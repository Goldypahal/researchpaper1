"""
Execution Trace Logger for Autonomous AI Research Agent.
Implements the graph schema defined in trace_schema.json:
- Nodes: Hypothesis, ExperimentSpec, ExecutionRun, MetricResult, ReflectiveAnalysis
- Edges: MOTIVATED_BY, IMPLEMENTED_AS, EVALUATED_IN, PRODUCED_METRIC, ANALYZED_BY, BRANCHED_FROM, ABANDONED_DEAD_END, BACKTRACKED_TO
- Computes: cycle_time, dead_end_ratio, path_to_first_improvement, backtracking_count.
"""

import json
import time
import os
from datetime import datetime

class TraceLogger:
    def __init__(self, experiment_id, agent_arm="Arm1_AutonomousAgent", model_base="SmallTransformerLM", total_gpu_budget_hours=10.0, baseline_loss=None):
        self.experiment_id = experiment_id
        self.agent_arm = agent_arm
        self.model_base = model_base
        self.total_gpu_budget_hours = total_gpu_budget_hours
        self.start_time = datetime.utcnow().isoformat() + "Z"
        self.baseline_loss = baseline_loss
        
        self.nodes = []
        self.edges = []
        self.incumbent_best_loss = float('inf')
        self.path_to_first_improvement = None
        self.dead_ends = 0
        self.backtracks = 0
        self.cycle_times = []

    def set_baseline_loss(self, loss):
        """Sets or updates the reference baseline loss for relative improvement calculations."""
        self.baseline_loss = float(loss)

    def add_hypothesis(self, iteration_idx, text, target_component, predicted_delta_loss, parent_hypo_id=None, provider=None):
        node_id = f"hypo_{iteration_idx}_{int(time.time()*1000)%100000}"
        node = {
            "node_id": node_id,
            "iteration_index": iteration_idx,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "node_type": "Hypothesis",
            "data": {
                "hypothesis_text": text,
                "target_component": target_component,
                "predicted_delta_loss": predicted_delta_loss,
                "provider": provider or "unknown"
            }
        }
        self.nodes.append(node)
        if parent_hypo_id:
            self.edges.append({
                "edge_id": f"e_{len(self.edges)+1}",
                "source_node_id": node_id,
                "target_node_id": parent_hypo_id,
                "relationship_type": "BRANCHED_FROM"
            })
        return node_id

    def add_experiment_spec(self, iteration_idx, hypo_id, code_patch):
        node_id = f"exp_{iteration_idx}_{int(time.time()*1000)%100000}"
        node = {
            "node_id": node_id,
            "iteration_index": iteration_idx,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "node_type": "ExperimentSpec",
            "data": {
                "code_patch": code_patch
            }
        }
        self.nodes.append(node)
        self.edges.append({
            "edge_id": f"e_{len(self.edges)+1}",
            "source_node_id": node_id,
            "target_node_id": hypo_id,
            "relationship_type": "MOTIVATED_BY"
        })
        return node_id

    def add_execution_run(self, iteration_idx, exp_id, status, exit_code, gpu_seconds, val_loss, train_loss):
        node_id = f"run_{iteration_idx}_{int(time.time()*1000)%100000}"
        
        rel_improvement = 0.0
        is_best = False
        if val_loss and val_loss < self.incumbent_best_loss:
            if self.incumbent_best_loss != float('inf'):
                rel_improvement = round(((self.incumbent_best_loss - val_loss) / self.incumbent_best_loss) * 100, 2)
            else:
                rel_improvement = 0.0
            self.incumbent_best_loss = val_loss
            is_best = True
            if self.path_to_first_improvement is None and rel_improvement > 0:
                self.path_to_first_improvement = iteration_idx

        node = {
            "node_id": node_id,
            "iteration_index": iteration_idx,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "node_type": "ExecutionRun",
            "data": {
                "execution_status": status,
                "exit_code": exit_code,
                "gpu_seconds_spent": gpu_seconds,
                "validation_loss": val_loss,
                "train_loss": train_loss,
                "relative_improvement_pct": rel_improvement,
                "is_incumbent_best": is_best
            }
        }
        self.nodes.append(node)
        self.edges.append({
            "edge_id": f"e_{len(self.edges)+1}",
            "source_node_id": node_id,
            "target_node_id": exp_id,
            "relationship_type": "IMPLEMENTED_AS"
        })
        return node_id, is_best

    def add_reflection(self, iteration_idx, run_id, hypo_id, reflection_text, decision, self_detected_failure=False, cycle_sec=0.0):
        node_id = f"refl_{iteration_idx}_{int(time.time()*1000)%100000}"
        node = {
            "node_id": node_id,
            "iteration_index": iteration_idx,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "node_type": "ReflectiveAnalysis",
            "data": {
                "agent_reflection": reflection_text,
                "branch_decision": decision,
                "self_detected_failure": self_detected_failure
            }
        }
        self.nodes.append(node)
        self.edges.append({
            "edge_id": f"e_{len(self.edges)+1}",
            "source_node_id": node_id,
            "target_node_id": run_id,
            "relationship_type": "ANALYZED_BY"
        })

        if decision == "REJECT_AND_BACKTRACK":
            self.dead_ends += 1
            self.backtracks += 1
            self.edges.append({
                "edge_id": f"e_{len(self.edges)+1}",
                "source_node_id": node_id,
                "target_node_id": hypo_id,
                "relationship_type": "ABANDONED_DEAD_END"
            })

        if cycle_sec > 0:
            self.cycle_times.append(cycle_sec)

        return node_id

    def export_trace(self, filepath):
        total_iterations = max([n["iteration_index"] for n in self.nodes], default=0)
        total_gpu_sec = sum([n["data"].get("gpu_seconds_spent", 0.0) for n in self.nodes])
        total_hypo = len([n for n in self.nodes if n["node_type"] == "Hypothesis"])
        dead_end_ratio = round(self.dead_ends / max(1, total_hypo), 3)
        avg_cycle = round(sum(self.cycle_times) / max(1, len(self.cycle_times)), 2)

        trace_data = {
            "experiment_metadata": {
                "experiment_id": self.experiment_id,
                "agent_arm": self.agent_arm,
                "model_base": self.model_base,
                "dataset_name": "SyntheticStateTransitionDyck",
                "total_gpu_budget_hours": self.total_gpu_budget_hours,
                "start_timestamp": self.start_time,
                "end_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "summary_statistics": {
                "total_iterations": total_iterations,
                "total_gpu_hours_used": round(total_gpu_sec / 3600.0, 4),
                "dead_end_ratio": dead_end_ratio,
                "path_to_first_improvement_steps": self.path_to_first_improvement or 0,
                "average_cycle_time_seconds": avg_cycle,
                "backtracking_count": self.backtracks,
                "baseline_loss": round(self.baseline_loss, 4) if self.baseline_loss is not None else None,
                "final_best_loss": round(self.incumbent_best_loss, 4) if self.incumbent_best_loss != float('inf') else None,
                "final_relative_improvement_pct": (
                    round(((self.baseline_loss - self.incumbent_best_loss) / self.baseline_loss) * 100, 2)
                    if (self.baseline_loss is not None and self.baseline_loss > 0 and self.incumbent_best_loss != float('inf'))
                    else 0.0
                )
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2)
        print(f"Trace log written to: {filepath} ({len(self.nodes)} nodes, {len(self.edges)} edges)")
        return trace_data
