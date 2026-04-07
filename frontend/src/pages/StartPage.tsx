/**
 * @file src/pages/StartPage.tsx
 * @description Primary start screen with branded header and workflow action buttons.
 */

import { AppButton } from "../components/ui/AppButton";
import { InteractiveTitle } from "../components/ui/InteractiveTitle";

interface StartPageProps {
  onAnalyzeOne: () => void;
  onAnalyzeMulti: () => void;
  onLoad: () => void;
  onHelp: () => void;
}

export function StartPage({ onAnalyzeOne, onAnalyzeMulti, onLoad, onHelp }: StartPageProps) {
  return (
    <main className="page-shell page-shell--landing start-page" style={{ position: 'relative' }}>
      <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 10 }}>
        <AppButton variant="primary" onClick={onHelp}>
          Help 
        </AppButton>
      </div>

      <section className="page-shell__content start-page__content">
        <header className="start-page__hero">
          <div className="start-page__branding">
            <InteractiveTitle text="Airline Analytics Capstone" className="start-page__title" />
            <p className="start-page__byline">by Blitz Analytics</p>
          </div>
        </header>

        <section className="start-page__actions">
          <div className="stacked-actions">
            <AppButton variant="primary" block onClick={onAnalyzeOne}>
              Analyze One Period
            </AppButton>
            <AppButton block onClick={onAnalyzeMulti}>
              Analyze Multiple Periods
            </AppButton>
            <AppButton block onClick={onLoad}>
              Load Data Set
            </AppButton>
          </div>
        </section>
      </section>
    </main>
  );
}
