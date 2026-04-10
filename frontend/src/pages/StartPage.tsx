/**
 * @file src/pages/StartPage.tsx
 * @description Primary start screen with branded header and workflow action buttons.
 */

import { useEffect, useRef, useState } from "react";
import { AppButton } from "../components/ui/AppButton";
import { InteractiveTitle } from "../components/ui/InteractiveTitle";

interface StartPageProps {
  onAnalyzeOne: () => void;
  onAnalyzeMulti: () => void;
  onLoad: () => void;
  onHelp: () => void;
  onAbout: () => void;
}

export function StartPage({ onAnalyzeOne, onAnalyzeMulti, onLoad, onHelp, onAbout }: StartPageProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      if (!menuOpen) return;
      const target = event.target as Node | null;
      if (!target) return;
      if (menuRef.current && !menuRef.current.contains(target)) {
        setMenuOpen(false);
      }
    }

    window.addEventListener("pointerdown", onPointerDown);
    return () => window.removeEventListener("pointerdown", onPointerDown);
  }, [menuOpen]);

  return (
    <main className="page-shell page-shell--landing start-page" style={{ position: 'relative' }}>
      <div
        ref={menuRef}
        className="start-page__menu"
        style={{ position: "absolute", top: "20px", right: "20px", zIndex: 30 }}
      >
        <AppButton
          variant="primary"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span className="start-page__menu-icon" aria-hidden="true">
            ☰
          </span>
        </AppButton>

        {menuOpen ? (
          <div className="start-page__menu-panel" role="menu" aria-label="Main menu">
            <button
              className="start-page__menu-item"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                onHelp();
              }}
            >
              Help
            </button>
            <button
              className="start-page__menu-item"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                onAbout();
              }}
            >
              About
            </button>
          </div>
        ) : null}
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
