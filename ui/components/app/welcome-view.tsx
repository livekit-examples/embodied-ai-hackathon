import { Button } from '@/components/ui/button';

function WelcomeImage() {
  return (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" role="presentation" className="mb-4 size-16">
        <rect x="4" y="5.75" width="16" height="14" stroke="currentcolor" strokeWidth="1.5"></rect>
        <line x1="21" y1="9.75" x2="21" y2="13.75" stroke="currentcolor" strokeWidth="2"></line>
        <line x1="12" y1="3.25" x2="12" y2="6.25" stroke="currentcolor" strokeWidth="2"></line>
        <line x1="3" y1="9.75" x2="3" y2="13.75" stroke="currentcolor" strokeWidth="2"></line>
        <rect x="9" y="11" width="0.5" height="2" fill="currentcolor" stroke="currentcolor" strokeWidth="1.5"></rect>
        <rect x="14.5" y="11" width="0.5" height="2" fill="currentcolor" stroke="currentcolor" strokeWidth="1.5"></rect>
    </svg>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <WelcomeImage />

        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          Chat live with your robotic desktop assistant
        </p>

        <Button
          size="lg"
          onClick={onStartCall}
          className="mt-6 w-64 rounded-full font-mono text-xs font-bold tracking-wider uppercase"
        >
          {startButtonText}
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <small>SPC Embodied AI Hackathon</small>
      </div>
    </div>
  );
};
