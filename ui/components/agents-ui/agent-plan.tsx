'use client';

import { useMemo } from 'react';
import { useTextStream } from '@livekit/components-react';
import { cn } from '@/lib/shadcn/utils';

const PLAN_TOPIC = 'agent.plan';

export type StepStatus = 'pending' | 'in_progress' | 'completed';
export type PlanPhase = 'idle' | 'planning' | 'executing' | 'completed';

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

  const showList = phase === 'executing' || phase === 'completed';

  return (
    <div className={cn('mx-auto w-full max-w-2xl pb-8 text-xl', className)}>
      {phase === 'planning' && <p>Planning{task ? `: ${task}` : '...'}</p>}
      {phase === 'completed' && <p className="font-semibold">Done!</p>}
      {showList && <AgentPlanList steps={steps} />}
    </div>
  );
}

function AgentPlanList({ steps }: { steps: PlanStep[] }) {
  return (
    <ol className="h-36 overflow-y-scroll scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100 bg-background rounded-md p-2 flex flex-col gap-2">
      {steps.map((step, index) => (
        <AgentPlanItem key={index} index={index + 1} {...step} />
      ))}
    </ol>
  );
}

function AgentPlanItem({
  index,
  description,
  status,
}: PlanStep & { index: number }) {
  return (
    <li className="flex items-baseline gap-2">
      <span className="text-foreground/60 font-mono min-w-8 text-right">{`${index}.`}</span>
      <span className="flex-1">{description}</span>
      <span className="opacity-20">({STATUS_LABEL[status]})</span>
    </li>
  );
}