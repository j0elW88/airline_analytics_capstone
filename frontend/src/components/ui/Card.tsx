/**
 * @file src/components/ui/Card.tsx
 * @description Reusable card container with optional header and actions area.
 */

import type { PropsWithChildren, ReactNode } from "react";

interface CardProps extends PropsWithChildren {
  title?: string;
  subtitle?: string;
  className?: string;
  headerRight?: ReactNode;
}

export function Card({ title, subtitle, className = "", headerRight, children }: CardProps) {
  return (
    <section className={`card ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="card__header">
          <div className="card__header-main">
            {title && <h3 className="card__title">{title}</h3>}
            {subtitle && <p className="card__subtitle">{subtitle}</p>}
          </div>
          {headerRight ? <div className="card__header-right">{headerRight}</div> : null}
        </header>
      )}
      <div className="card__body">{children}</div>
    </section>
  );
}





