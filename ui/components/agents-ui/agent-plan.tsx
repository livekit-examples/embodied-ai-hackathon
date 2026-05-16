'use client';

import { memo, useEffect, useMemo, useRef } from 'react';
import { useTextStream } from '@livekit/components-react';
import scrollIntoView from 'smooth-scroll-into-view-if-needed';
import { Fade } from '@/components/agents-ui/blocks/agent-session-view-01/components/agent-session-block';
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
    <div className="relative">
      <ol className="h-36 overflow-y-auto [&::-webkit-scrollbar]:hidden [scrollbar-width:none] bg-background rounded-md px-2 py-6 flex flex-col gap-2">
        {steps.map((step, index) => (
          <AgentPlanItem key={index} index={index + 1} {...step} />
        ))}
      </ol>
      <Fade top className="absolute inset-x-0 top-0 z-10 h-6 rounded-t-md" />
      <Fade bottom className="absolute inset-x-0 bottom-0 z-10 h-6 rounded-b-md" />
    </div>
  );
}

const AgentPlanItem = memo(function AgentPlanItem({
  index,
  description,
  status,
}: PlanStep & { index: number }) {
  const isActive = status === 'in_progress';
  const ref = useRef<HTMLLIElement>(null);

  useEffect(() => {
    if (!isActive || !ref.current) return;
    scrollIntoView(ref.current, {
      behavior: 'smooth',
      block: 'center',
      duration: 1200,
      ease: (t) => t * t * (3 - 2 * t),
    });
  }, [isActive]);

  return (
    <li
      ref={ref}
      className={cn(
        'flex items-baseline gap-2 transition-opacity duration-1000 ease-in-out will-change-[opacity]',
        isActive ? 'opacity-100' : 'opacity-40'
      )}
    >
      <span className="font-mono min-w-8 text-right">{`${index}.`}</span>
      <span className="flex-1">{description}</span>
      <span className="opacity-20">({STATUS_LABEL[status]})</span>
    </li>
  );
});