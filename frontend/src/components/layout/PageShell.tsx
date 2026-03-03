/**
 * @file src/components/layout/PageShell.tsx
 * @description Consistent page frame with title/subtitle and body content slots.
 */

import type { PropsWithChildren } from "react";

interface PageShellProps extends PropsWithChildren {
  title: string;
  subtitle?: string;
  landing?: boolean;
  actions?: React.ReactNode;
}

export function PageShell({ title, subtitle, landing = false, actions, children }: PageShellProps) {
  return (
    <main className={`page-shell ${landing ? "page-shell--landing" : ""}`}>
      <section className="page-shell__content">
        <header className="page-shell__header">
          <div>
            <h1>{title}</h1>
            {subtitle ? <p>{subtitle}</p> : null}
          </div>
          {actions ? <div className="page-shell__actions">{actions}</div> : null}
        </header>
        {children}
      </section>
    </main>
  );
}





