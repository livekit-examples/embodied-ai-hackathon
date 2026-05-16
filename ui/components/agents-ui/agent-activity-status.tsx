'use client';

import { useMemo } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useTextStream } from '@livekit/components-react';
import { Shimmer } from '@/components/ai-elements/shimmer';
import { cn } from '@/lib/shadcn/utils';

const ACTIVITY_TOPIC = 'agent.activity';

const ACTIVITY_LABELS: Record<string, string> = {
  planning: 'Planning',
  executing: 'Executing',
};

export interface AgentActivity {
  activity: string;
  detail: string;
}

const EMPTY_ACTIVITY: AgentActivity = { activity: '', detail: '' };

/**
 * Subscribes to the `agent.activity` text stream published by the orchestrator
 * agent's `activity_status` helper and returns the latest `{activity, detail}`
 * status. Each agent-side `send_text` call produces one entry in `textStreams`
 * whose `text` field is the full JSON payload of that stream, so reading the
 * tail of the array yields the current status.
 */
export function useAgentActivity(): AgentActivity {
  const { textStreams } = useTextStream(ACTIVITY_TOPIC);
  return useMemo(() => {
    const latest = textStreams[textStreams.length - 1];
    if (!latest) return EMPTY_ACTIVITY;
    try {
      const parsed = JSON.parse(latest.text) as Partial<AgentActivity>;
      return {
        activity: parsed.activity ?? '',
        detail: parsed.detail ?? '',
      };
    } catch {
      return EMPTY_ACTIVITY;
    }
  }, [textStreams]);
}

function formatActivity(activity: string, detail: string): string {
  const label = ACTIVITY_LABELS[activity] ?? activity;
  return detail ? `${label}: ${detail}` : label;
}

/**
 * Renders the agent's current activity (e.g. "Planning: hand me the hammer")
 * sourced from the `agent.activity` text stream published by the orchestrator
 * agent's `activity_status` helper.
 *
 * Renders nothing when no activity is set. Uses `AnimatePresence` with
 * `mode="wait"` so the previous label fully exits before the new one enters,
 * yielding a clean crossfade on phase changes.
 */
export function AgentActivityStatus({ className }: { className?: string }) {
  const { activity, detail } = useAgentActivity();
  const label = activity ? formatActivity(activity, detail) : '';

  return (
    <AnimatePresence mode="wait" initial={false}>
      {label && (
        <motion.div
          key={label}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
          className={cn(
            'pointer-events-none mx-auto block w-full max-w-2xl pb-4 text-center text-sm font-semibold',
            className
          )}
        >
          <Shimmer duration={2}>{label}</Shimmer>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
