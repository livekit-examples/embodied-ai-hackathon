'use client';

import { AnimatePresence, motion } from 'motion/react';
import { useAgent } from '@livekit/components-react';
import { Shimmer } from '@/components/ai-elements/shimmer';
import { cn } from '@/lib/shadcn/utils';

const ACTIVITY_LABELS: Record<string, string> = {
  planning: 'Planning',
  executing: 'Executing',
};

function formatActivity(activity: string, detail: string): string {
  const label = ACTIVITY_LABELS[activity] ?? activity;
  return detail ? `${label}: ${detail}` : label;
}

/**
 * Renders the agent's current activity (e.g. "Planning: hand me the hammer") sourced
 * from the `agent.activity` and `agent.activity.detail` participant attributes
 * published by the orchestrator agent's `activity_status` helper.
 *
 * Renders nothing when no activity is set. Uses `AnimatePresence` with
 * `mode="wait"` so the previous label fully exits before the new one enters,
 * yielding a clean crossfade on phase changes.
 */
export function AgentActivityStatus({ className }: { className?: string }) {
  const agent = useAgent();
  const activity = agent.attributes?.['agent.activity'] ?? '';
  const detail = agent.attributes?.['agent.activity.detail'] ?? '';
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
