/**
 * @file src/pages/HomePage.tsx
 * @description Home screen that routes users to primary app workflows.
 */

import { PageShell } from "../components/layout/PageShell";
import { AppButton } from "../components/ui/AppButton";

interface HomePageProps {
  onHistory: () => void;
  onLoaded: () => void;
  onStart: () => void;
}

export function HomePage({ onHistory, onLoaded, onStart }: HomePageProps) {
  return (
    <PageShell
      title="Airline Market Analysis Dashboard"
      subtitle="Modular React workflow with reusable components"
      landing
    >
      <section className="home-actions">
        <AppButton onClick={onHistory}>History</AppButton>
        <AppButton onClick={onLoaded}>Loaded Data Sets</AppButton>
        <AppButton variant="primary" onClick={onStart}>
          Start
        </AppButton>
      </section>
    </PageShell>
  );
}





