'use client';

import { useMemo } from 'react';
import { useTextStream } from '@livekit/components-react';
import { cn } from '@/lib/shadcn/utils';

const PLAN_TOPIC = 'agent.plan';

export type StepStatus = 'pending' | 'in_progress' | 'completed';
export type PlanPhase = 'idle' | 'planning' | 'executing';

export interface PlanStep {
  description: string;
  status: StepStatus;
}

export interface AgentPlan {
  phase: PlanPhase;
  task: string;
  steps: PlanStep[];
}

const EMPTY_PLAN: AgentPlan = { phase: 'idle', task: '', steps: [] };

/**
 * Subscribes to the `agent.plan` text stream published by the orchestrator
 * agent's `PlanProgress` helper and returns the latest snapshot. The agent
 * sends a full snapshot on each update, so the tail of `textStreams` is the
 * current state.
 */
export function useAgentPlan(): AgentPlan {
  const { textStreams } = useTextStream(PLAN_TOPIC);
  return useMemo(() => {
    const latest = textStreams[textStreams.length - 1];
    if (!latest) return EMPTY_PLAN;
    try {
      const parsed = JSON.parse(latest.text) as Partial<AgentPlan>;
      return {
        phase: parsed.phase ?? 'idle',
        task: parsed.task ?? '',
        steps: parsed.steps ?? [],
      };
    } catch {
      return EMPTY_PLAN;
    }
  }, [textStreams]);
}

const STATUS_LABEL: Record<StepStatus, string> = {
  pending: 'pending',
  in_progress: 'in progress',
  completed: 'done',
};

/**
 * Renders the agent's current plan and per-step status as a simple numbered
 * list, sourced from the `agent.plan` text stream. Renders nothing while
 * the agent is idle. Styling is intentionally minimal for now.
 */
export function AgentPlanView({ className }: { className?: string }) {
  const { phase, task, steps } = useAgentPlan();
  if (phase === 'idle') return null;

  return (
    <div className={cn('mx-auto w-full max-w-2xl pb-4 text-sm', className)}>
      {phase === 'planning' && <p>Planning{task ? `: ${task}` : '...'}</p>}
      {phase === 'executing' && (
        <ol className="list-decimal pl-6">
          {steps.map((s, i) => (
            <li key={i}>
              {s.description}{' '}
              <span className="opacity-60">({STATUS_LABEL[s.status]})</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
